#pragma once

#include <utility>

#include "value.h"

namespace ai {

template <typename T>
class Trace {
    public:
        Trace(const Value<T>& root)
        {
        }

    private:
        void build(const Value<T>* v) {
            if (!nodes_.contains(v)) {
                nodes_.insert(v);
                for (const Value<T>* c : v.children()) {
                    edges_.insert({c, v});
                    build(c);
                }
            }
        }

    private:
        std::set<const Value<T>*> nodes_{};
        std::set<std::pair<const Value<T>*, const Value<T>*>> edges_{};
};

}
