#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include <iostream>

struct bell {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  }
};

int main() {
  std::cout << cudaq::contrib::draw(bell{}) << '\n';
}