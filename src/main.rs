use burn::tensor::Tensor;
use burn::backend::WgpuBackend;

type Backend = WgpuBackend;

fn main() {
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]]);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    println!("{}", tensor_1 + tensor_2)
}
