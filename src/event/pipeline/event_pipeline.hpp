#ifndef EYE_TRACK_EVENT_PIPELINE_HPP
#define EYE_TRACK_EVENT_PIPELINE_HPP
#include "event/pipeline/event_pipeline.h"
namespace eye_track {
/* C++ façade for starting the C event pipeline from an abstract input source. */
class EventPipeline {
public:
    /* Connects the selected source to the globally managed task pipeline. */
    static bool start(const event_source_t &source /* Callbacks and state used to read events. */)
    {
        return event_pipeline_start(&source);
    }
};
} // namespace eye_track
#endif
