
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;
using namespace py::literals;

namespace ovms {

class PythonBackend {
    py::module_ pyovmsModule;
    py::object tensorClass;

public:
    bool initialize();
    bool deinitialize(); // always true
    py::object createOvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, const std::string& datatype, py::ssize_t size);

};


} // namespace ovms