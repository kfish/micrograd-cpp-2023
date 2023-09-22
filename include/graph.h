#pragma once

#include <iomanip>
#include <utility>

#include "value.h"

namespace ai {

template <typename T>
class Trace {
    public:
        Trace(const Value<T>& root)
        {
            build(root);
        }

        const std::set<RawValue<T>*>& nodes() const {
            return nodes_;
        }

        const std::set<std::pair<RawValue<T>*, RawValue<T>*>> edges() const {
            return edges_;
        }

    private:
        void build(const Value<T>& v) {
            if (!nodes_.contains(v.get())) {
                nodes_.insert(v.get());
                for (auto && c : v->children()) {
                    edges_.insert({c.get(), v.get()});
                    build(c);
                }
            }
        }

    private:
        std::set<RawValue<T>*> nodes_{};
        std::set<std::pair<RawValue<T>*, RawValue<T>*>> edges_{};
};

template <typename T>
class NodeName {
    public:
        NodeName(const RawValue<T>* ptr)
            : ptr_(ptr)
        {}

        const RawValue<T>* get() const {
            return ptr_;
        }

    private:
        const RawValue<T>* ptr_;
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const NodeName<T>& node) {
    return os << "\"node" << node.get() << "\"";
}

template <typename T>
class NodeOp {
    public:
        NodeOp(const RawValue<T>* ptr)
            : ptr_(ptr)
        {}

        const RawValue<T>* get() const {
            return ptr_;
        }

    private:
        const RawValue<T>* ptr_;
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const NodeOp<T>& node) {
    return os << "\"node" << node.get() << node.get()->op() << "\"";
}

template <typename T>
class Graph {
    public:
        //Graph(const Value<T>& root)
        Graph(const std::shared_ptr<RawValue<T>>& root)
            : trace_(root)
        {
        }

        std::ostream& dump(std::ostream& os) const {
            auto old_precision = os.precision();

            os << "digraph G {\n"
               << "  rankdir = \"LR\";"
               << std::endl;

            os << std::fixed << std::setprecision(4);

            for (const RawValue<T>* node : trace_.nodes()) {
                // For any value in the graph, create a rectangular ("record") node
                // for it
                os << "  " << NodeName<T>(node)
                   << " [label = \"{ " << node->label()
                   << " | data=" << node->data()
                   << " | grad=" << node->grad()
                   << " }\", shape=\"record\"]"
                   << std::endl;

                if (node->op()) {
                    // If this value is the result of an operation, create
                    // an op node for it
                    os << "  " << NodeOp<T>(node)
                       << " [label = \"" << node->op() << "\"]"
                       << std::endl;

                    // And connect the op to it
                    os << "  " << NodeOp<T>(node)
                       << " -> " << NodeName<T>(node) << ";"
                       << std::endl;
                }
            }

            // Edges
            for (auto && [n1, n2] : trace_.edges()) {
                // Connect n1 to the op node of n2
                os << "  " << NodeName<T>(n1) << " -> " << NodeOp<T>(n2) << ";" << std::endl;
            }

            os << "}" << std::endl;

            os << std::setprecision(old_precision);

            return os;
        }

    private:
        Trace<T> trace_;
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const Graph<T>& graph) {
    return graph.dump(os);
}

}
