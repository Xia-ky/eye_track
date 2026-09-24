/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "cli/cli_router.hpp"

#include "common/kavl.h"
#include "common/queue.h"
#include "log/log.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <new>
#include <regex>
#include <string>
#include <string_view>

namespace eye_track::cli {
namespace {

constexpr std::size_t max_tokens = 12U;
constexpr std::size_t max_input_size = 128U;

struct ParameterType;
struct Node;
struct CommandRecord;

TAILQ_HEAD(ParameterList, ParameterType);
TAILQ_HEAD(NodeList, Node);
TAILQ_HEAD(CommandList, CommandRecord);

struct ParameterType {
    std::string name; /* Placeholder spelling used to select this argument type. */
    std::regex expression; /* Compiled validation rule for captured command tokens. */
    TAILQ_ENTRY(ParameterType) link; /* Intrusive-list links for registered parameter types. */

    ParameterType(std::string_view parameter_name,
                  std::string_view parameter_expression)
        : name(parameter_name),
          expression(parameter_expression.begin(), parameter_expression.end())
    {
    }
};

struct Node {
    std::string token; /* Literal token or placeholder spelling represented by this tree node. */
    ParameterType *parameter = nullptr; /* Registered regex type when this node is a placeholder. */
    Node *literal_root = nullptr; /* AVL root containing literal-token children. */
    NodeList placeholder_children; /* Ordered list of regex-validated placeholder children. */
    const cli_command_definition_t *command = nullptr; /* Command terminating at this node, if any. */
    KAVL_HEAD(Node) avl; /* Intrusive AVL metadata for literal-token lookup. */
    TAILQ_ENTRY(Node) placeholder_link; /* Intrusive link for placeholder-child membership. */

    explicit Node(std::string_view node_token = {},
                  ParameterType *node_parameter = nullptr)
        : token(node_token), parameter(node_parameter)
    {
        TAILQ_INIT(&placeholder_children);
        std::memset(&avl, 0, sizeof(avl));
    }
};

struct CommandRecord {
    cli_command_definition_t definition; /* Owned copy referenced by terminal tree nodes. */
    TAILQ_ENTRY(CommandRecord) link; /* Intrusive link for router-owned command definitions. */
};

int cli_node_compare(const Node *left, const Node *right)
{
    /* Keep literal-token siblings ordered so AVL lookup remains logarithmic. */
    return left->token.compare(right->token);
}

KAVL_INIT2(cli_node, static inline, Node, avl, cli_node_compare)

struct TokenList {
    std::array<std::string_view, max_tokens> values{}; /* Non-owning slices into normalized input. */
    std::size_t size = 0U; /* Number of valid entries in values. */
};

struct NormalizedInput {
    std::array<char, max_input_size> storage{}; /* Bounded buffer holding collapsed-whitespace input. */
    std::string_view value{}; /* View into storage covering the normalized command. */
};

bool normalize_input(std::string_view input, NormalizedInput &normalized)
{
    std::size_t used = 0U; /* Number of normalized bytes written into storage. */
    bool pending_space = false; /* Whether one separator is due before the next token byte. */

    /* Collapse runs of spaces while preserving one separator between tokens. */
    for (const char value /* Current input byte being normalized. */ : input) {
        if (value == ' ') {
            pending_space = used != 0U;
            continue;
        }
        if (pending_space) {
            if (used + 1U >= normalized.storage.size()) {
                return false;
            }
            normalized.storage[used++] = ' ';
            pending_space = false;
        }
        if (used + 1U >= normalized.storage.size()) {
            return false;
        }
        normalized.storage[used++] = value;
    }
    normalized.storage[used] = '\0';
    normalized.value = std::string_view(normalized.storage.data(), used);
    return true;
}

bool split_tokens(std::string_view text, TokenList &tokens)
{
    std::size_t cursor = 0U; /* Current position in text while finding token boundaries. */

    tokens.size = 0U;
    while (cursor < text.size()) {
        while (cursor < text.size() && text[cursor] == ' ') {
            ++cursor;
        }
        if (cursor == text.size()) {
            break;
        }
        if (tokens.size == tokens.values.size()) {
            return false;
        }
        const std::size_t start = cursor; /* First byte of the next token slice. */
        while (cursor < text.size() && text[cursor] != ' ') {
            ++cursor;
        }
        tokens.values[tokens.size++] = text.substr(start, cursor - start);
    }
    return tokens.size != 0U;
}

bool is_placeholder(std::string_view token)
{
    return token.size() >= 3U && token.front() == '_' && token.back() == '_';
}

Node *find_literal(Node *parent, std::string_view token)
{
    Node probe(token); /* Temporary comparison key; it is never inserted into the tree. */
    return kavl_find(cli_node, parent->literal_root, &probe, nullptr);
}

ParameterType *find_parameter(ParameterList *parameters,
                              std::string_view name)
{
    ParameterType *parameter; /* Current registered type inspected by the lookup. */
    TAILQ_FOREACH(parameter, parameters, link) {
        if (parameter->name == name) {
            return parameter;
        }
    }
    return nullptr;
}

Node *find_placeholder(Node *parent, ParameterType *parameter)
{
    Node *node; /* Current placeholder child inspected under this parent. */
    TAILQ_FOREACH(node, &parent->placeholder_children, placeholder_link) {
        if (node->parameter == parameter) {
            return node;
        }
    }
    return nullptr;
}

void destroy_children(Node *parent)
{
    while (parent->literal_root != nullptr) {
        Node *node = kavl_erase_first(cli_node, &parent->literal_root); /* Detached literal child being destroyed. */
        destroy_children(node);
        delete node;
    }
    while (!TAILQ_EMPTY(&parent->placeholder_children)) {
        Node *node = TAILQ_FIRST(&parent->placeholder_children); /* First placeholder child removed from its list. */
        TAILQ_REMOVE(&parent->placeholder_children, node, placeholder_link);
        destroy_children(node);
        delete node;
    }
}

bool regex_matches(const ParameterType &parameter, std::string_view value)
{
    return std::regex_match(value.begin(), value.end(), parameter.expression);
}

const cli_command_definition_t *match_command(
    Node *parent, const TokenList &tokens, std::size_t index,
    cli_invocation_t &invocation)
{
    if (index == tokens.size) {
        return parent->command;
    }

    /* Prefer literal paths so exact commands take precedence over placeholders. */
    if (Node *literal = find_literal(parent, tokens.values[index])) {
        if (const auto *command =
                match_command(literal, tokens, index + 1U, invocation)) {
            return command;
        }
    }

    Node *placeholder; /* Candidate regex node tested against the current token. */
    TAILQ_FOREACH(placeholder, &parent->placeholder_children,
                  placeholder_link) {
        if (invocation.argument_count >= CLI_MAX_ARGUMENTS ||
            !regex_matches(*placeholder->parameter, tokens.values[index])) {
            continue;
        }
        const std::size_t argument_index = invocation.argument_count++; /* Slot reserved for this captured token. */
        invocation.arguments[argument_index] = {
            tokens.values[index].data(), tokens.values[index].size()
        };
        if (const auto *command =
                match_command(placeholder, tokens, index + 1U, invocation)) {
            return command;
        }
        invocation.argument_count = argument_index;
    }
    return nullptr;
}

} // namespace

struct Router::Impl {
    Node root; /* Root of the complete command token tree. */
    ParameterList parameters; /* Owned regex definitions referenced by placeholder nodes. */
    CommandList commands; /* Owned command definitions referenced by terminal nodes. */
    std::size_t command_count = 0U; /* Number of registered command definitions. */

    Impl()
    {
        TAILQ_INIT(&parameters);
        TAILQ_INIT(&commands);
    }

    ~Impl()
    {
        destroy_children(&root);
        while (!TAILQ_EMPTY(&commands)) {
            CommandRecord *record = TAILQ_FIRST(&commands);
            TAILQ_REMOVE(&commands, record, link);
            delete record;
        }
        while (!TAILQ_EMPTY(&parameters)) {
            ParameterType *parameter = TAILQ_FIRST(&parameters);
            TAILQ_REMOVE(&parameters, parameter, link);
            delete parameter;
        }
    }
};

Router::Router() : impl_(std::make_unique<Impl>())
{
    if (!register_parameter("_STRING_", "^\\S+$") ||
        !register_parameter("_UINT_", "^[0-9]+$")) {
        throw std::bad_alloc();
    }
}

Router::~Router() = default;

bool Router::register_parameter(std::string_view placeholder,
                                std::string_view expression)
{
    if (!is_placeholder(placeholder) || expression.empty() ||
        find_parameter(&impl_->parameters, placeholder) != nullptr) {
        return false;
    }
    try {
        auto *parameter = new ParameterType(placeholder, expression); /* Compiled regex owned by the router. */
        TAILQ_INSERT_TAIL(&impl_->parameters, parameter, link);
        return true;
    } catch (const std::exception &) {
        LOG_ERROR("cli_router", "parameter allocation failed\r\n");
        return false;
    }
}

bool Router::register_command(const cli_command_definition_t &command)
{
    if (command.pattern == nullptr || command.handler == nullptr) {
        return false;
    }

    if (command.pattern[0] == '\0') {
        if (impl_->root.command != nullptr) {
            return false;
        }
        try {
            auto *record = new CommandRecord{command, {nullptr, nullptr}}; /* Stable owned empty-command definition. */
            TAILQ_INSERT_TAIL(&impl_->commands, record, link);
            impl_->root.command = &record->definition;
            ++impl_->command_count;
            return true;
        } catch (const std::exception &) {
            LOG_ERROR("cli_router", "empty command allocation failed\r\n");
            return false;
        }
    }

    /* Split the declared pattern once so each token can be inserted into the tree. */
    TokenList tokens; /* Bounded sequence of literal and placeholder token views. */
    if (!split_tokens(command.pattern, tokens)) {
        return false;
    }

    Node *parent = &impl_->root; /* Current prefix node extended by the next pattern token. */
    try {
        for (std::size_t index = 0U; index < tokens.size; ++index) {
            const std::string_view token = tokens.values[index]; /* Pattern component inserted at this depth. */
            if (is_placeholder(token)) {
                ParameterType *parameter = /* Registered regex needed to create or reuse the placeholder edge. */
                    find_parameter(&impl_->parameters, token);
                if (parameter == nullptr) {
                    return false;
                }
                Node *node = find_placeholder(parent, parameter); /* Existing placeholder edge, if already registered. */
                if (node == nullptr) {
                    node = new Node(token, parameter);
                    TAILQ_INSERT_TAIL(&parent->placeholder_children, node,
                                      placeholder_link);
                }
                parent = node;
            } else {
                Node *node = find_literal(parent, token); /* Existing literal edge, if already registered. */
                if (node == nullptr) {
                    node = new Node(token);
                    Node *existing = /* Existing equal key returned when insertion finds a duplicate. */
                        kavl_insert(cli_node, &parent->literal_root, node,
                                    nullptr);
                    if (existing != node) {
                        delete node;
                        node = existing;
                    }
                }
                parent = node;
            }
        }
        if (parent->command != nullptr) {
            return false;
        }
        auto *record = new CommandRecord{command, {nullptr, nullptr}}; /* Stable owned definition linked to the router. */
        TAILQ_INSERT_TAIL(&impl_->commands, record, link);
        parent->command = &record->definition;
        ++impl_->command_count;
        return true;
    } catch (const std::exception &) {
        LOG_ERROR("cli_router", "command node allocation failed: %s\r\n",
                  command.pattern);
        return false;
    }
}

cli_process_result_t Router::process(std::string_view input, char *output,
                                     std::size_t output_size)
{
    /* Validate the destination before writing any response bytes. */
    if (output == nullptr || output_size == 0U) {
        return CLI_PROCESS_DONE;
    }
    output[0] = '\0';

    NormalizedInput normalized; /* Storage plus view for collapsed-space command text. */
    if (!normalize_input(input, normalized)) {
        std::snprintf(output, output_size,
                      "Command too long.\r\n");
        return CLI_PROCESS_DONE;
    }

    TokenList tokens; /* Token slices used to traverse the command tree. */
    cli_invocation_t invocation{}; /* Match result passed to the selected handler. */
    invocation.input = {normalized.value.data(), normalized.value.size()};
    if (normalized.value.empty()) {
        if (impl_->root.command != nullptr) {
            return impl_->root.command->handler(output, output_size,
                                                &invocation,
                                                impl_->root.command->context);
        }
    } else if (split_tokens(normalized.value, tokens)) {
        if (const auto *command = /* Terminal command definition matching the complete token path. */
                match_command(&impl_->root, tokens, 0U, invocation)) {
            return command->handler(output, output_size, &invocation,
                                    command->context);
        }
    }

    std::snprintf(output, output_size,
                  "Command not recognised. Enter 'help' to list commands.\r\n");
    return CLI_PROCESS_DONE;
}

bool Router::write_help(char *output, std::size_t output_size) const
{
    /* Reject unusable buffers before initializing the help response. */
    if (output == nullptr || output_size == 0U) {
        return false;
    }

    std::size_t used = 0U; /* Number of help bytes already written to output. */
    output[0] = '\0';
    CommandRecord *record; /* Current owned command definition in registration order. */
    TAILQ_FOREACH(record, &impl_->commands, link) {
        const char *help = record->definition.help; /* Help string belonging to this command. */
        if (help == nullptr) {
            continue;
        }
        const std::size_t length = std::strlen(help); /* Help bytes copied before CRLF. */
        if (length + 2U >= output_size - used) {
            return false;
        }
        std::memcpy(output + used, help, length);
        used += length;
        output[used++] = '\r';
        output[used++] = '\n';
        output[used] = '\0';
    }
    return true;
}

std::size_t Router::command_count() const noexcept
{
    return impl_->command_count;
}

} // namespace eye_track::cli
