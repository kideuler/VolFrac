# VolFracAI

This project is built using CMake and uses GoogleTest for unit testing. Follow the steps below to build, test, and run results as configured in the GitHub Actions workflow.

## Prerequisites

- CMake (version 3.10 or higher)
- A C++ compiler (e.g., g++ or clang++)
- Git
- Python 3.8 (with venv support)
- Make

## Building and Running Locally

The following instructions mirror the steps defined in the workflow:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/VolFrac.git
   cd VolFrac
   ```

2. **Set up Python Environment**
    ```
    sudo apt install python3-venv        # On Ubuntu; on Mac, Python3 is usually pre-installed
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install numpy
    pip install torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install scikit-learn
    ```

3. **Build and Install GoogleTest with `GLIBCXX_USE_CXX11_ABI` off**
    ```
    git clone https://github.com/google/googletest.git -b release-1.12.1
    cd googletest
    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=/location/to/install -DCMAKE_BUILD_TYPE=Release ..
    make
    sudo make install
    cd ../..
    rm -rf googletest
    ```

4. **Activate Environment and Configure CMake**
    ```
    source venv/bin/activate
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    ```

5. **Build and Run Tests**
    ```
    cmake --build build --parallel
    cd build
    make test
    make data
    ```

6. **Train Model**
    ```
    python py/Train_Curvature.py
    ```

7. **Test Model**
    ```
    make results
    ```