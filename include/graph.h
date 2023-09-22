#pragma once

#include <utility>

#include "value.h"

namespace ai {

template <typename T>
class Trace {
    public:
        Trace(const Value<T>& root)
        {
            build(&root);
        }

        const std::set<const Value<T>*>& nodes() const {
            return nodes_;
        }

        const std::set<std::pair<const Value<T>*, const Value<T>*>> edges() const {
            return edges_;
        }

    private:
        void build(const Value<T>* v) {
            if (!nodes_.contains(v)) {
                nodes_.insert(v);
                for (const Value<T>* c : v->children()) {
                    edges_.insert({c, v});
                    build(c);
                }
            }
        }

    private:
        std::set<const Value<T>*> nodes_{};
        std::set<std::pair<const Value<T>*, const Value<T>*>> edges_{};
};

template <typename T>
class NodeName {
    public:
        NodeName(const Value<T>* value)
            : value_(value)
        {}

    private:
        template <typename U>
        friend std::ostream& operator<<(std::ostream& os, const NodeName<U>&);

        const Value<T>* value_;
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const NodeName<T>& node) {
    return os << "\"node" << node.value_ << "\"";
}

template <typename T>
class NodeOp {
    public:
        NodeOp(const Value<T>* value)
            : value_(value)
        {}

    private:
        template <typename U>
        friend std::ostream& operator<<(std::ostream& os, const NodeOp<U>&);

        const Value<T>* value_;
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const NodeOp<T>& node) {
    return os << "\"node" << node.value_ << node.value_->op() << "\"";
}

template <typename T>
class Graph {
    public:
        Graph(const Value<T>& root)
            : trace_(root)
        {
            Trace<T> trace(root);
        }

        std::ostream& dump(std::ostream& os) const {
            os << "digraph G {\n"
               << "  rankdir = \"LR\";"
               << std::endl;

            for (const Value<T>* node : trace_.nodes()) {
                // For any value in the graph, create a rectangular ("record") node
                // for it
                os << "  " << NodeName(node)
                   << " [label = \"{ data=" << node->data() << " }\", shape=\"record\"]"
                   << std::endl;

                if (node->op()) {
                    // If this value is the result of an operation, create
                    // an op node for it
                    os << "  " << NodeOp(node)
                       << " [label = \"" << node->op() << "\"]"
                       << std::endl;

                    // And connect the op to it
                    os << "  " << NodeOp(node)
                       << " -> " << NodeName(node) << ";"
                       << std::endl;
                }
            }

            // Edges
            for (auto && [n1, n2] : trace_.edges()) {
                // Connect n1 to the op node of n2
                os << "  " << NodeName(n1) << " -> " << NodeOp(n2) << ";" << std::endl;
            }

            os << "}" << std::endl;

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
