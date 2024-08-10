## Candle Tensor Operations Cheatsheet

| ops | pytorch | candle |
| --- | ------------- | ---------|
| **Tensor Creation**| `torch.tensor([1,2],[3,4])` | `Tensor::new(&[[1u32,2],[3,4]],&Device::Cpu)` |
| | `torch.ones((2,4))`|`Tensor::ones((2,4),DType::F32, &Device::Cpu)` |
| | `torch.zeros((2,4))`|`Tensor::zeros((2,4),DType::F32, &Device::Cpu)` |
| | `torch.ones_like(tensor)`|`Tensor::ones_like(&Tensor)` |
| | `torch.rand((2,4))`| `Tensor::rand(0.0f32, 1.0, (2,4), &Device::Cpu)` |
| | `torch.randn((2,4))`|`Tensor::randn(1.0f32, 5.0, (2,4), &Device::Cpu)` |
| | `torch.rand_like(tensor)`|`Tensor::rand_like(&Tensor, 0.0 , 1.0)` |
| |`torch.arange(start=1.0, end=10.0)` |`Tensor::arange(1.0, 10.0, &Device::Cpu)` |
| |`torch.arange(start=1.0, end=10.0, step=0.5)` |`Tensor::arange_step(1.0, 10.0, 0.5, &Device::Cpu)` |
| | | |
| **Arithmetic**| | |
| Matmul | `torch.bmm(tensor, tensor)`|`Tensor::matmul(&Tensor, &Tensor)` |
| | `torch.matmul(tensor, tensor)`|`Tensor::broadcast_matmul(&Tensor, &Tensor)` |
| Add | `torch.add(tensor, tensor)`|`Tensor::add(&Tensor, &Tensor)` |
| | |`Tensor::broadcast_add(&Tensor, &Tensor)` |
| Sub |`torch.sub(tensor, tensor)` |`Tensor::sub(&Tensor, &Tensor)` |
| | |`Tensor::broadcast_sub(&Tensor, &Tensor)` |
| Mul |`torch.mul(tensor, tensor)` |`Tensor::mul(&Tensor, &Tensor)` |
| | |`Tensor::broadcast_mul(&Tensor, &Tensor)` |
| Div |` torch.div(tensor, tensor)` |`Tensor::div(&Tensor, &Tensor)` |
| | |` Tensor::broadcast_div(&Tensor, &Tensor)` |
| Exp |`torch.exp(tensor)` |`Tensor::exp(&Tensor)` |
| Sqrt |`torch.sqrt(tensor)` |`Tensor::sqrt(&Tensor)` |
| Abs |`torch.abs(tensor)` |`Tensor::abs(&Tensor)` |
| Sum |`torch.sum(tensor, dim=1)` |`Tensor::sum(&Tensor, 1)` |
| |`torch.sum(tensor, dim=1, keepdim=True)` |`Tensor::sum_keepdim(&Tensor, 1)` |
| |`torch.sum(tensor)` |`Tensor::sum_all(&Tensor)` |
| Mean |`torch.mean(tensor, dim=1)` |`Tensor::mean(&Tensor, 1)` |
| |`torch.mean(tensor, dim=1, keepdim=True) ` |`Tensor::mean_keepdim(&Tensor, 1)` |
| |`torch.mean(tensor)` |`Tensor::mean_all(&Tensor)` |
| Max |`torch.max(tensor, dim=1)` |`Tensor::max(&Tensor, 1)` |
| |`torch.max(tensor, dim=1, keepdim=True)` |`Tensor::max_keepdim(&Tensor, 1)` |
| |`torch.minimum(tensor, tensor)` |`Tensor::broadcast_maximum(&Tensor)` |
| | |`Tensor::maximum(&Tensor, &Tensor/Scalar)` |
| Min |`torch.min(tensor, dim=1)` |`Tensor::min(&Tensor, 1)` |
| |`torc.min(tensor, dim=1, keepdim=True) ` |`Tensor::min_keepdim(&Tensor, 1)` |
| |`torch.minimum(tensor, tensor)` |`Tensor::broadcast_minimum(&Tensor, &Tensor)` |
| | |`Tensor::minimum(&Tensor, &Tensor/Scalar)` |
| Cumsum |`torch.cumsum(tensor, dim=1)` |`Tensor::cumsum(&Tensor, 1)` |
| | | |
| **Comparision** |  |  |
| a <= b |`torch.le(tensor, tensor)` |`Tensor::le(&Tensor, &Tensor)` |
| | |`Tensor::broadcast_le(&Tensor, &Tensor)` |
| a < b |`torch.lt(tensor, tensor)` |`Tensor::lt(&Tensor, &Tensor)` |
| | |`Tensor::broadcast_lt(&Tensor, &Tensor)` |
| a >= b |`torch.ge(tensor, tensor)` |`Tensor::ge(&Tensor, &Tensor)` |
| | |`Tensor::broadcast_ge(&Tensor, &Tensor)` |
| a > b |`torch.gt(tensor, tensor)` |`Tensor::gt(&Tensor, &Tensor)` |
| | |`Tensor::broadcast_gt(&Tensor, &Tensor)` |
| a == b |`torch.eq(tensor, tensor)` |`Tensor::eq(&Tensor, &Tensor)` |
| ||`Tensor::broadcast_eq(&Tensor, &Tensor)`|
| a != b |`torch.ne(tensor, tensor)` |`Tensor::ne(&Tensor, &Tensor)` |
| | |`Tensor::broadcast_ne(&Tensor, &Tensor)` |
| | | |
| **Other**| | |
| Reshape |`tensor.reshape((-1,2))` |`Tensor.reshape(((),2)) ` |
| Cat |`torch.cat((tensor, tensor), dim=0)` |`Tensor::cat(&[&Tensor, &Tensor], 0) ` |
| Stack |`torch.stack((tensor, tensor), dim=0)` |`Tensor::stack(&[&Tensor, &Tensor], 0)` |
| Index-select |`torch.index_select(tensor, dim=0, index=tensor)` |`Tensor::index_select(&Tensor, &Tensor, 0)` |
| Slice |`tensor[:,0:5,1]` |`Tensor.i((..,0..5,1))` |
| Eye |`torch.eye(6)` |`Tensor::eye(6, DType::F32, &Device::Cpu)` |
| Full |`torch.full((2,4), 3.14159)` |`Tensor::full(3.14159, (2,4), &Device::Cpu)` |
| ~~Tril~~ |`torch.tril(tensor, diagonal=0)`|`Tensor::tril2(usize, DType, &Device)`|
| |`torch.triu(tensor, diagonal=0)`|`Tensor::triu2(usize, DType, &Device)`|
| Ceil |`torch.ceil(tensor)`|`Tensor::ceil(&Tensor)`|
| Clamp |`torch.clamp(tensor, min=0.0, max=1.0)`|`Tensor::clamp(&Tensor, 0.0, 1.0)`|
| |`torch.clamp(tensor, mid=tensor, max=tensor)`|`Tensor::clamp(&Tensor, &Tensor, &Tensor)`|
| Broadcast |`tensor.expand((2,2,4))` |`Tensor.expand((2,2,4))` |
| |`torch.broadcast_to(tensor, shape=(2,2,4))` |`Tensor::broadcast_as(&Tensor, (2,2,4))` |
| Gather |`torch.gather(tensor, dim=1, index=tensor)` |`Tensor::gather(&Tensor, &Tensor, 1)` |
| Floor |`torch.floor(tensor)` |`Tensor::floor(&Tensor)` |
| Narrow |`torch.narrow(tensor, dim, start, length)` |`Tensor::narrow(&Tensor, dim, start, length)` |
| Permute |`torch.permute(tensor, (0,2,1))` |`Tensor::permute(&Tensor, (0,2,1))` |
| Pow |`torch.pow(tensor, tensor)` |`Tensor::pow(&Tensor, &Tensor)` |
| |`torch.pow(tensor, 3.0)` |`Tensor::powf(&Tensor, 3.0)` |
| Repeat |`tensor.repeat((4,2,1))` |`Tensor.repeat((4,2,1))` |
| Transpose |`torch.transpose(tensor, (2,1))` |`Tensor::transpose(&Tensor, 2, 1)` |
| |`torch.t(tensor)` |`Tensor::t(&Tensor)` |
| Where|`torch.where(tensor>0, on_true=tensor, on_false=tensor)` |`Tensor::where_cond(&Tensor, &Tensor, &Tensor)` |
| | | |



