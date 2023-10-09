#include "python_backend.hpp"
#include <iostream>

namespace py = pybind11;
using namespace py::literals;
using namespace ovms;

bool PythonBackend::initialize() {
    py::initialize_interpreter();
    py::gil_scoped_acquire acquire;
    py::exec(R"(
        import sys
        print("Python version:")
        print (sys.version)
    )");

    try {
    pyovmsModule = py::module_::import("pyovms");
    tensorClass = pyovmsModule.attr("Tensor");
    } catch (const pybind11::error_already_set& e) {
        std::cout << "PythonBackend initialization failed: " << e.what() << std::endl;
        return false;
    } catch (std::exception& e) {
        std::cout << "PythonBackend initialization failed: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool PythonBackend::deinitialize() {
    py::gil_scoped_acquire acquire;
    py::finalize_interpreter();
    return true;
}


py::object PythonBackend::createOvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, const std::string& datatype, py::ssize_t size) {
    py::gil_scoped_acquire acquire;
    // TO DO: Error handling
    return tensorClass.attr("create_from_data")(name, ptr, shape, datatype, size);
}
