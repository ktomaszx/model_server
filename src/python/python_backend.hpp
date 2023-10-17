#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace ovms {

struct PyObjectWrapper {
    py::object obj;

    PyObjectWrapper() = delete;
    PyObjectWrapper(PyObjectWrapper& other) = delete;
    PyObjectWrapper(const py::object& obj);
    PyObjectWrapper(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, const std::string& datatype, py::ssize_t size);
    ~PyObjectWrapper();

    template <typename T> 
    T getProperty(const std::string& name) const {
        py::gil_scoped_acquire acquire;
        try {
            std::cout << "PyObjectWrapper object: " <<
                std::endl << "ptr - " << obj.ptr() <<
                std::endl << "ref_count - " << obj.ref_count() << std::endl;

            std::cout << "PythonObjectWrapper::getting property: " << name << std::endl;
            py::object myObj = obj.attr("name");
            std::cout << "dupa";
            std::string retVal1 = obj.attr("namess").cast<std::string>();
            std::cout << "PythonObjectWrapper::getting property hardcoded successful";
            T retVal = obj.attr(name.c_str()).cast<T>();
            std::cout << "PythonObjectWrapper::getting property dynamic successful";
            return retVal;
        } catch (const pybind11::error_already_set& e) {
            std::cout << "PythonObjectWrapper::getProperty failed: " << e.what() << std::endl;
            throw e;
        } catch (std::exception& e) {
            std::cout << "PythonObjectWrapper::getProperty failed: " << e.what() << std::endl;
            throw e;
        }
    }
};

class PythonBackend {
    //py::module_ pyovmsModule;
    //py::object tensorClass;

public:
    py::module_ pyovmsModule;
    py::object tensorClass;
    PythonBackend();
    ~PythonBackend();
    static bool createPythonBackend(PythonBackend** pythonBackend);
    bool createOvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, const std::string& datatype, 
                                  py::ssize_t size, std::unique_ptr<PyObjectWrapper>& outTensor);

};


} // namespace ovms