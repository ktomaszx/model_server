#include "python_backend.hpp"
#include <iostream>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace ovms;


PyObjectWrapper::PyObjectWrapper(const py::object& other) {
    py::gil_scoped_acquire acquire;
    obj = other;
    std::cout << "PyObjectWrapper constructor: " <<
    std::endl << "ptr - " << obj.ptr() <<
    std::endl << "ref_count - " << obj.ref_count() << std::endl;
};

PyObjectWrapper::PyObjectWrapper(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, 
                                 const std::string& datatype, py::ssize_t size) {
    py::gil_scoped_acquire acquire;
    std::cout << "PyObjectWrapper constructor from data " << std::endl;
    py::module_ pyovmsModule_ = py::module_::import("pyovms");
    py::object tensorClazz = pyovmsModule_.attr("Tensor");
    obj = std::move(tensorClazz.attr("create_from_data")(name, ptr, shape, datatype, size));
    std::cout << "PyObjectWrapper constructor from data end " << std::endl;
}

PyObjectWrapper::~PyObjectWrapper() {
    py::gil_scoped_acquire acquire;
    std::cout << "PyObjectWrapper destructor: " <<
    std::endl << "ptr - " << obj.ptr() <<
    std::endl << "ref_count - " << obj.ref_count() << std::endl;
    obj.dec_ref();
    std::cout << "PyObjectWrapper destructor end " << std::endl;
}
/*
template <typename T> 
T PyObjectWrapper::getProperty<T>(const std::string& name) {
    py::gil_scoped_acquire acquire;
    try {
        return obj.attr(name).cast<T>();
    } catch (const pybind11::error_already_set& e) {
        std::cout << "PythonObjectWrapper::getProperty faild: " << e.what() << std::endl;
        throw e;
    } catch (std::exception& e) {
        std::cout << "PythonObjectWrapper::getProperty failed: " << e.what() << std::endl;
        throw e;
    }
}
*/

bool PythonBackend::createPythonBackend(PythonBackend* pythonBackend) {
    py::gil_scoped_acquire acquire;
    try {
    pythonBackend = new PythonBackend();
    std::cout << pythonBackend;
    } catch (const pybind11::error_already_set& e) {
        std::cout << "PythonBackend initialization failed: " << e.what() << std::endl;
        return false;
    } catch (std::exception& e) {
        std::cout << "PythonBackend initialization failed: " << e.what() << std::endl;
        return false;
    }
    return true;
}

PythonBackend::PythonBackend() {
    py::gil_scoped_acquire acquire;
    py::print("Creating python backend");
    pyovmsModule = py::module_::import("pyovms");
    tensorClass = pyovmsModule.attr("Tensor");
}

PythonBackend::~PythonBackend() {
    py::gil_scoped_acquire acquire;
    py::print("Removing python backend");
    tensorClass.dec_ref();
    pyovmsModule.dec_ref();
}

bool PythonBackend::createOvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, 
                                             const std::string& datatype, py::ssize_t size, std::unique_ptr<PyObjectWrapper>& outTensor) {
    py::gil_scoped_acquire acquire;
    try {
        py::print("Gonna create_from_data");
        outTensor = std::make_unique<PyObjectWrapper>(name, ptr, shape, datatype, size);
        py::print("Gonna return");
        return true;
    } catch (const pybind11::error_already_set& e) {
        std::cout << "PythonBackend::createOvmsPyTensor - Py Error: " << e.what();
        return false;
    } catch (std::exception& e) {
        std::cout << "PythonBackend::createOvmsPyTensor - Error: " << e.what();
        return false;
    }
    // TO DO: Error handling
}
