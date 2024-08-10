
use candle_core::{ DType, Device, IndexOp, Result, Tensor};


#[allow(warnings)]
fn main() ->Result<()> {

    println!("Hello from Chapter 0");
    let dtype = DType::F32;
    let device = Device::Cpu;

    // ------------------------------------- Tensor Creation ----------------------------------------------//
    
    let tensor = Tensor::new(&[1u32,2,3,4],&device)?;
    let _tensor = Tensor::new(&[1.0f32,2.0,3.0], &device)?;
    // println!("tensor: {}", tensor);
    /*
        note :: 1u32 specifies the DType , bydefault it takes i32 which is not supported.
                1.0f32 specifies the F32 precision, bydefault it takes f64. 
                supported DTypes =>  i64, u8, u32, f32, f64
                
        output => [1, 2, 3, 4] Tensor[[4], u32]
    */

    let ones_tensor = Tensor::ones((2, 4), dtype, &device)?;
    let zeros_tensor = Tensor::zeros((2, 4), dtype, &device)?;
    // println!("ones: {}\nzeros: {}",ones_tensor, zeros_tensor)
    /*
        note :: default to f32 precision.

        output => [[1., 1., 1., 1.],
                  [1., 1., 1., 1.]] Tensor[[2, 4], f32]
        creates a new tensor filled with ones/zeros.
    */
    
    let _tensor = Tensor::zeros_like(&_tensor)?;
    let _tensor = Tensor::ones_like(&tensor)?;
    // println!("ones_like: {}",_tensor)
    /*
        output => [0.0, 0.0, 0.0, 0.0] Tensor[[4], f32]
        creates a new tensor filled with ones/zeros with same shape, dtype, and device as the input tensor. 
    */

    let rand_tensor = Tensor::rand(0.0f32, 1.0, (4,4), &device)?;
    let randn_tensor = Tensor::randn(1.0, 3.0, (2,8), &device)?;
    let randlike_tensor = Tensor::randn_like(&ones_tensor, 0.0, 4.0)?;
    // println!("rand_tensor: {}\nrandn_tensor: {}\nrandlike_tensor: {}", rand_tensor,randn_tensor, randlike_tensor);
    /*
        rand => args: lower bound , upper bound , shape , device.
                returns the tensor of specified shape with random values between upper and lower bound.
        
        randn => args: mean , standard deviation , shape , device
                returns the tensor of specified shape with random values having mean and standard deviation as specified.

        randn_like => args: Tensor, mean , standard deviation 
                    returns the tensor of input_tensor shape with random values having mean and standard deviation as specified.
    
    */

    let arange_tensor = Tensor::arange(0u32, 10, &device)?;
    let arange_step_tensor = Tensor::arange_step(0.0, 1.0, 0.05, &device)?;
    // println!("arange_tensor: {}\narange_step_tensor: {}", arange_tensor, arange_step_tensor);
    /*
        arange_tensor => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        returns tensor ranging from start to end with default step of 1.
                        
        arange_step_tensor => [0.0000, 0.2000, 0.4000, 0.6000, 0.8000]
        returns tensor ranging from start to end with specified steps.                    
    */
    
    // --------------------------------------------------- End ---------------------------------------------------------------\\

    // _____________________________________________________________________________________________________________
    // |---------------------------------------- Arithmetic --------------------------------------------------------//

    // ------------------------------------- Data Preparation -----------------------------------------------------||
    
    let a = Tensor::new(&[
            [ 0u32,  1 ],
            [ 4,  5],
            [ 8,  9] ],&device)?; // [3, 2]

    let b = Tensor::new(&[
            [ 2u32, 3, 6],
            [ 7, 10, 11]    ],&device)?; // [2, 3]]

    let c = Tensor::new(&[
            [ 2u32,  3 ],
            [ 6,  7],
            [ 10,  11] ],&device)?; // [3, 2]

    let d = Tensor::new(&[2.0f32], &device)?; // [1] for broadcasting purpose
    let e = Tensor::new(&[1u32, 2], &device)?; // [3] for broadcasting purpose
    
    // ---------------------------------------- Arithmetic Ops ------------------------------------------------------//
    
    let matmul = Tensor::matmul(&a.to_dtype(DType::F32)?, &b.to_dtype(DType::F32)?)?;
    let _matmul = a.to_dtype(DType::F32)?.matmul(&b.to_dtype(DType::F32)?)?;
    // println!("matmul: {}", matmul);
    /*
        note :: matmul operation doens't support u32 , i64 DTypes.

        matmul => [[  7.,  10.,  11.], [ 43.,  62.,  79.], [ 79., 114., 147.]] S=(3,3)
                dimensions of lhs tensor (B, T) and rhs tensor (T, C) must be in this format to apply matmul 
                in n dimensions the lhs tensors last dim must match with rhs tensors second last dim. 
                eg. x(32,512,768) @ y(32,768,2) = z(32,512,2)
                    (B, T, C)   @    (B, C, D) =    (B, T, D)
    */  

    let add = Tensor::add(&a, &c)?;
    let _add = a.add(&c)?;
    let _add = (&a + &c)?;
    let broadcast_add = a.broadcast_add(&e)?;
    // println!("add: {}\nbroadcast_add: {}",add,broadcast_add);
    /*
        add => [[ 2,  4],[10, 12],[18, 20]] S=(3,2) 
        - elementwise addition of two tensors (both must be of same size).

        broadcast_add => [[ 1,  3],[5, 7],[9, 11]] S=(3,2)
        - addition of given input tensors elements with provided tensor.         (last dim must be same of other tensor).
        - eg.  x(4,2,4) + y(4)   = z(4,2,4)                                      (last dim either be same or 1 ).
               x(4,2,4) + y(1,4) = z(4,2,4) / x(4,2,4) + y(2,4) = z(4,2,4)       (at dim -2 shape should either be 1 or same as the input dimension).
    
    */

    // all these operations have same explanation as above |^
    let sub = Tensor::sub(&c, &a)?;
    let sub = Tensor::broadcast_sub(&c, &e)?;
    // println!("sub: {}",sub);
    /*
        note :: sub operations on u32 DType will cause errors when resulting answers are negative 
                either convert them to i64 or f32/f64 before operations.

        let a = Tensor::new(&[[1f32, 2, 4],[4, 2, 1]], &device)?;
        let b = Tensor::new(&[[0f32, 6, 3],[0f32, 6, 3]], &device)?;
        a.sub(&b)?;

        output => [[ 1., -4.,  1.],[ 1., -4.,  1.]] S=(2,3) 
        - elementwise subtraction of two tensors (both must be of same size).

        let c = Tensor::new(&[[3.0f32, 2.0, 3.0]], &device)?;
        a.broadcast_sub(&c)?;

        output => [[-2.,  0.,  1.], [ 1.,  0., -2.]] S=(2,3)
            subtraction of given input tensors elements with provided tensor.      (last dim must be same of other tensor).
            eg.  x(4,2,4) - y(4)   = z(4,2,4)                                      (last dim either be same or 1 ).
                 x(4,2,4) - y(1,4) = z(4,2,4) / x(4,2,4) + y(2,4) = z(4,2,4)       (at dim -2 shape should either be 1 or same as the input dimension).
    */

    let mul = ( &a * &c )? ;
    let _mul = a.broadcast_mul(&e)?;
    // println!("mul: {}",mul);
    /*
        inputs are used from Data Preparation block.
        a*c
        mul => [[ 0,  3], [24, 35], [80, 99]]  S=(3,2) 
            elementwise multiplication of two tensors (both must be of same size).

        a.broadcast_mul(&e)
        broadcast_mul => [[ 0,  2], [4, 10], [8, 18]] S=(3,2)
            multiplication of given input tensors elements with provided tensor.        (last dim must be same of other tensor).
            eg.  x(4,2,4) * y(4)   = z(4,2,4)                                           (last dim either be same or 1 ).
                 x(4,2,4) * y(1,4) = z(4,2,4) / x(4,2,4) + y(2,4) = z(4,2,4)            (at dim -2 shape should either be 1 or same as the input dimension).
    
    */

    let div = Tensor::div(&a.to_dtype(DType::F32)?, &c.to_dtype(DType::F32)?)?;
    let _div = a.broadcast_div(&e)?;
    // println!("div: {}",div);
    /*
        inputs are used from Data Preparation block
        note :: DType U32 division won't keep decimal places, to keep decimal places we have to convert u32 to f32.

        a.f32/c.f32 
        div => [[0.0000, 0.3333], [0.6667, 0.7143], [0.8000, 0.8182]] S=(3,2) 
            elementwise division of two tensors (both must be of same size).

        a.broadcast_div(&e)
        broadcast_div => [[ 0,  0],[4, 2],[8, 4]] S=(3,2)
            division of given input tensors elements with provided tensor.        (last dim must be same of other tensor).
            eg.  x(4,2,4) / y(4)   = z(4,2,4)                                     (last dim either be same or 1 ).
                 x(4,2,4) / y(1,4) = z(4,2,4) / x(4,2,4) + y(2,4) = z(4,2,4)      (at dim -2 shape should either be 1 or same as the input dimension).
    
    */
    
    let exp = Tensor::exp(&d)?;
    // println!("exp: {}",exp);
    /* 
        exp => [7.3891]
        returns a new tensor with the exponential of the elements of the input tensor.
    */

    let sqrt = Tensor::sqrt(&a.to_dtype(dtype)?)?;
    // println!("sqrt: {}",sqrt);
    /* 
        note :: sqrt doesn't support the u8, u32, i64 DTypes.

        sqrt => [[0.0000, 1.0000], [2.0000, 2.2361], [2.8284, 3.0000]] S=(3,2)
        returns a new tensor with the square-root of the elements.
    */
    
    let sample = Tensor::new(&[-1.933f32, 3.44, -4.424], &device)?;
    let abs = sample.abs()?;
    // println!("abs: {}",abs);
    /* 
        abs => [1.9330, 3.4400, 4.4240]
        returns the absolute value of each element
    */

    let sum = a.sum(1)?;
    let _sum = a.sum_keepdim(1)?;
    let __sum = a.sum_all()?;
    // println!("sum: {}",sum);
    /* 
        inputs are used from Data Preparation block.

        sum => [ 1,  9, 17] S=(3)
        returns the sum of elements in the specified dimension squeezing that dimension.

        sum_keepdim => [[1],  [9], [17]] S=(3,1)
        returns the sum of elements in the specified dimension keeping that dimension.

        sum_all => [27] S=()/scalar
        returns the sum of all elements on all dimensions.
    */

    let mean = a.to_dtype(DType::F32)?.mean(1)?;
    // println!("mean: {}",mean);
    /* 
        inputs are used from Data Preparation block.
        note :: mean don't work on u32 , i64 DTypes but won't raise an error 
                returns tensor of zeros instead , convert to f32 to avoid.

        mean => [0.5000, 4.5000, 8.5000] S=(3)
        returns the mean value of all elements in the specified dimension sqeezing that dimension.

        mean_keepdim => [[0.5000], [4.5000], [8.5000]] S=(3,1)
        returns the mean value of all elements in the specified dimension keeping the dimension.

        mean_all => [4.5000] S=()/scalar
    */

    let sample = Tensor::new(&[5u32],&device)?;

    let max = a.max(1)?;
    let _max = a.max_keepdim(1)?;
    let maximum = a.maximum(&c)?;
    let broadcast_maximum = a.broadcast_maximum(&sample)?;
    // println!("max: {}\nmaximum: {}",max, maximum);
    /* 
        
        max => [1, 5, 9] S=(3)
        returns the maximum value of all elements in the specified dimension sqeezing that dimension.

        max_keepdim => [[1], [5], [9]] S=(3,1)
        returns the maximum value of all elements specified dimension keeping the dimension.

        maximun => [[ 2,  3], [ 6,  7], [10, 11]] S=(3,2)
        computes the element-wise maximum between input/rhs and other/lhs tensor and returns the maximum values tensor from both 
        lhs/other can be tensor or scalar value.

        broadcast_maximum => [[5, 5], [5, 5], [8, 9]]
        division of given input tensors elements with provided tensor.              (last dim must be same of other tensor).
        eg.  x(4,2,4) max    y(4)   = z(4,2,4)                                      (last dim either be same or 1 ).
                x(4,2,4) max y(1,4) = z(4,2,4) / x(4,2,4) + y(2,4) = z(4,2,4)       (at dim -2 shape should either be 1 or same as the input dimension).
    */
    
    let min = a.min(1)?;
    let _min = a.min_keepdim(1)?;
    let minimun = a.minimum(5.0f32)?;
    let broadcast_minimum = a.broadcast_minimum(&d.to_dtype(DType::U32)?)?;
    // println!("min: {}\nminimun: {}",min, minimun);
    /*
        min => [0, 4, 8] S=(3)
        returns the maximum value of all elements in the specified dimension sqeezing that dimension.

        min_keepdim => [[0], [4], [8]] S=(3,1)
        returns the maximum value of all elements specified dimension keeping the dimension.

        minimum => [[ 2,  3], [ 6,  7], [10, 11]] S=(3,2)
        computes the element-wise maximum between input/rhs and other/lhs tensor and returns the maximum values tensor from both 
        lhs/other can be tensor or scalar value.

        broadcast_minimum => [[0, 1], [4, 5], [5, 5]]
        division of given input tensors elements with provided tensor.             (last dim must be same of other tensor).
        eg.  x(4,2,4) min    y(4)   = z(4,2,4)                                     (last dim either be same or 1 ).
                x(4,2,4) min y(1,4) = z(4,2,4) / x(4,2,4) + y(2,4) = z(4,2,4)      (at dim -2 shape should either be 1 or same as the input dimension).

    */

    let sample = Tensor::new(&[[1f32, 2., 3., 4.],[5., 6., 7., 8.]], &device)?;
    let cumsum = a.to_dtype(DType::F32)?.cumsum(1)?;
    // println!("cumsum: {}", cumsum);
    /* 
        note :: cumsum only supports float DType (f32, f64)
        cumsum => [[ 1.,  3.,  6., 10.], [ 5., 11., 18., 26.]] S=(2,4)
        returns the cumulative sum of elements in the specified dimension.
    */

    // ------------------------------------------------- End ---------------------------------------------------------------\\

    // --------------------------------------------- Comparision ----------------------------------------------------------//
    
    // ------------------------------------------- Data Preparation -------------------------------------------------------||
    
    let a = Tensor::new(&[
        [ 0u32,  1 ],
        [ 4,  5],
        [ 10,  11] ],&device)?; // [3, 2]


    let b = Tensor::new(&[
        [ 2u32,  3 ],
        [ 6,  7],
        [ 8,  9] ],&device)?; // [3, 2]

    let c = Tensor::new(&[2.0f32], &device)?; 

// ------------------------------------------------- Comparision Ops -------------------------------------------------------//

    let le = a.le(&b)?;
    let lt = b.lt(6.5)?;
    // println!("le: {}\nlt: {}", le, lt);
    /* 
        note :: all comparison operations accepts tensor/scalar value as other/rhs argument
                , but if passed tensor (other/rhs) it should be same shape as the input/lhs tensor.
                can pass scalar value as other/rhs argument if want to check against one value. 
                can check tensor of any DType with any scalar value (f32, f64, u32, u8, i64).

        le => [[1, 1], [1, 1], [0, 0]] s=(3,2)
        element-wise comparison with lower-equal, the returned tensor uses value 1 where self <= rhs and 0 otherwise.

        lt => [[1, 1], [1, 0], [0, 0]]
        element-wise comparison with lower-than, the returned tensor uses value 1 where self < rhs and 0 otherwise.
    */

    let ge = a.ge(&b)?;
    let gt = b.gt(8.0)?;
    // println!("ge: {} gt: {}", ge, gt);
    /* 
        ge => [[0, 0], [0, 0], [1, 1]]
        element-wise comparison with greater-equal, the returned tensor uses value 1 where self >= rhs and 0 otherwise.

        gt => [[0, 0], [0, 0], [0, 1]]
        element-wise comparison with greater-than, the returned tensor uses value 1 where self > rhs and 0 otherwise.
    */

    let eq = a.eq(4.0)?;
    let ne = a.ne(4.0)?;
    // println!("eq: {}\nne: {}", eq, ne);
    /* 
        eq => [[0, 0], [1, 0], [0, 0]]
        element-wise equality (lhs.i == rhs.i).

        ne => [[1, 1], [0, 1], [1, 1]]
        element-wise non-equality (lhs.i != rhs.i).
    */
    // ------------------------------------------------- End ---------------------------------------------------------------\\


    // ------------------------------------------------- Other -------------------------------------------------------------//

    // ------------------------------------------- Data Preparation -------------------------------------------------------||
    
    let a = Tensor::new(&[
        [ 0f32,  1. ],
        [ -2.,  9.],
        [ 8.,  -7.] ], &device)?; // [3, 2]

    let b = Tensor::new(&[
        [ 2u32, 3, 4, 6],
        [ 7, 8, 9, 11]], &device)?; // [2, 4]]
    
    let c = Tensor::new(&[[ 0u32,  1 ], [ 4,  5], [ 10,  11] ], &device)?; // [3, 2]
    let d = Tensor::new(&[[ 2u32, 3], [ 7, 8 ], [10, 11] ], &device)?; // [3, 2]
    let e = Tensor::new(&[2.0f32], &device)?;  // [1]
    let idx = Tensor::new(&[0u32,0,1,0], &device)?; // [4]

    // -------------------------------------------------- Other Ops --------------------------------------------------------//


    let neg = a.neg()?;
    // println!("neg: {}", neg);
    /* 
        note :: only supports f32, f64 DTypes.

        neg => [[-0., -1.], [2., 9.], [-8., 7.]]
        multiply each element in the tensor with -1 (sign invert) and returns new tensor.  
    */

    let reshape = b.reshape((2,(),2))?;
    // println!("reshape: {}", reshape);
    /* 
        note :: this single implementation consists of view() & reshape() functionality of pytorch.
                if the input tensor is contiguous then view() is applied
                otherwise reshape() is used and the return tensor is contiguous.
                () can be used to represent the -1 of pytorch. 

        reshape => [[[ 2,  3],[ 4,  6]],
                    [[ 7,  8],[ 9, 11]]] S=(2,2,2)
        returns the tensor with reshaped dimensions. 
    */

    let cat = Tensor::cat(&[&c,&d], 1)?;
    // println!("cat: {}", cat);
    /* 
        note :: input tensors must be of same size and same DType. 

        cat => [[ 0,  1,  2,  3],
                [ 4,  5,  7,  8],
                [10, 11, 10, 11]] , //[3, 4]
        concatenates two or more tensors along a particular dimension.
    */

    let stack = Tensor::stack(&[&c,&d], 1)?;
    // println!("stack: {}", stack);
    /* 
        note :: input tensors must be of same size and same DType. 

        stack =>[[[ 0,  1], [ 2,  3]],
                [[ 4,  5], [ 7,  8]],
                [[10, 11], [10, 11]]] , //[3,2,2]
        stacks two or more tensors along a particular dimension.
    */

    let index_select = c.index_select(&idx, 1)?;
    // println!("index_select: {}", index_select);
    /* 
        note :: indexes tensor must be of u32/i64 DType .
                values inside indexes tensors should not be greater than maximum index of input tensor.
                eg. a[2,4] then index tensor should have values between 0-1 if dim = 0 and 0-3 if dim = 1.

        index-select =>[[ 0,  0,  1,  0],
                        [ 4,  4,  5,  4],
                        [10, 10, 11, 10]]  //[3,4]
        select values for the input tensor at the target indexes across the specified dimension.
    */

    let slice = c.i((..,..1))?;
    // println!("slice: {}",slice);
    /* 
        note :: equivalent to pytorchs tensor[:,1].
                [:,:1,4:] can be represented as (..,..1,4..).

        slice => [1, 5, 11] //[3]
        returns a slicing iterator which are the chunks of data necessary to reconstruct the desired tensor.
    */

    let eye = Tensor::eye(6, dtype, &device)?;
    // println!("eye: {}", eye);
    /* 
        eye => [[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]]  // [3,3]
        returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
    */

    let full = Tensor::full(12.0, (2,4), &device)?;
    // println!("full: {}", full);
    /* 
        note :: returned tensor is not contiguous.

        full => [[12., 12.],
                 [12., 12.]] // [2,2]
        returns a new tensor with all the elements having the same specified value.
    */

    let tril = Tensor::tril2(4, dtype, &device)?;
    // println!("tril: {}", tril);
    /* 
        tril => [[1., 0., 0.],
                 [1., 1., 0.],
                 [1., 1., 1.]]  // [3,3]
    */

    let triu = Tensor::triu2(6, dtype, &device)?;
    // println!("triu: {}", triu);
    /* 
        triu => [[1., 1., 1.],
                 [0., 1., 1.],
                 [0., 0., 1.]]  // [3,3]
    */
    
    let sample = Tensor::new(&[[1.0f32, -0.66, 0.01], [-3., 20., 0.99]], &device)?;
    let ceil = sample.ceil()?;
    // println!("ceil: {}", ceil);
    /* 
        ceil => [[ 1., -0.,  1.],
                 [-3., 20.,  1.]]  // [2,3]
        returns nearest greater integer value if value is inbetween -0.99 to 0.99 if integer value then returns as it is.
    */

    let clamp = sample.clamp(0.0f32, 1.0)?;
    // println!("clamp: {}", clamp);
    /* 
        note :: if passed scaler values only single value is required
                if passed tensor then tensor must be of same size as input tensor.
                it will replace values from given tensors (at same index) when codintion meets of cutoff.

        clamp => [[1.0000, 0.0000, 0.0100],
                  [0.0000, 1.0000, 0.9900]]  // [2,3]
        returns tensor with values being cut off to min/max value if value goes beyond range of min/max otherwise returns same value.
    */

    let broadcast = sample.broadcast_as((2,2,3))?;
    // println!("broadcast: {}", broadcast);
    /* 
        note :: shape sizes last dims must be same as input tensors dims.
                extra dimensions should be added on 0 dim only.
                eg. a[2,4] s(32,2,4)✅  s(2,4,32)⛔️ s(2,32,4)⛔️

        broadcast => [[[1.0000e0, -6.6000e-1,  1.0000e-2],
                       [-3.0000e0,   2.0000e1,  9.9000e-1]],
                      [[1.0000e0, -6.6000e-1,  1.0000e-2],
                       [-3.0000e0,   2.0000e1,  9.9000e-1]]]  //[2,2,3]
        returns tensor with extra added dimensions on 0 dim.
        same as stacking input tensor on dim 0 multiple times
    */
    let idx = Tensor::new(&[[1u32,0],[0,2]], &device)?;
    let gather = c.gather(&idx, 0)?;
    // println!("gather: {}", gather);
    /* 
        note :: index-select and gather do the same thing but on different levels.
                index-select takes 1d tensor input for selecting indexes.
                whereas gather expects index tensor to be same size as input tensor.
        
        gather => [[ 4,  1],
                   [ 0, 11]]  //[2,2]
        selects values of input tensor according to specified index tensor.
    */

    let sample = Tensor::new(&[1.0, 4.5, 6.3, -3.3, 5., -0.44], &device)?;
    let floor = sample.floor()?;
    // println!("floor: {}", floor);
    /* 
        floor => [ 1.,  4.,  6., -4.,  5., -1.] // 6
        returns tensor with nearest small integer if float value otherwise same value for integers.
    */
    
    let sample = Tensor::new(&[[1u32, 2, 3], [4, 5, 6], [7, 8, 9]], &device)?;
    let narrow = sample.narrow(0, 1, 2)?;   // row wise
    let _narrow = sample.narrow(1, 1, 2)?;  // column wise
    // println!("narrow: {}", narrow);
    /* 
        narrow =>(dim=0,start=1,len=2) 
                  [[4, 5, 6],
                   [7, 8, 9]]  //[2,3] row wise
                
                (dim=1,start=1,len=2) 
                  [[2, 3],
                   [5, 6],
                   [8, 9]]  //[3,2] column wise
        returns tensor which will be of values selected from specified dim,
        start will be the first index(row/column),
        len will select values alongside the dim and index till (start+len).

        eg. x[3,4] d=0 , s=1, l=2 => x[1:3,:] (pytorch equivalent indexing).
            x[3,4] d=1 , s=1, l=2 => x[:,1:3] (pytorch equivalent indexing).
    */
    
    let sample = Tensor::rand(0.0, 1.0, (2,4,6), &device)?;
    let permute = sample.permute((2,1,0))?;
    // println!("permute: {:?}", permute);
    /*  
        note :: dims index should not exceed original dimension size 
                eg a[2,4,6] dims=3 input_dim=(1,0,2) below 3 input_dim=(0,1,3) ⛔️ 

        permute => [6,4,2]
        returns tensor with dimensions permutation.
    */

    let sample = Tensor::new(&[2.0,3.,4.], &device)?; 
    let e = Tensor::new(&[2.0,3.0,4.], &device)?; 
    let pow = sample.pow(&e)?; 
    // println!("pow: {}", pow);
    /* 
        note :: supports only to f32/f64 DTypes.
                expect input tensor and exponent tensor should be of same DType.

        pow => [4.0000,  27.0000, 256.0000] //[3]
        returns tensor with input tensor elements power raised according with exponent tensor elements.
    */
    
    let sample = Tensor::new(&[2.0, 3.0, 4.0], &device)?;
    let powf = sample.powf(3.0)?;
    // println!("powf: {}", powf);
    /* 
        note :: supports only on f32/f64 DTypes.

        powf => [8., 27., 64.]  // [3]
        returns tensor with elements raised to power of e (x^e).
    */
    
    let sample = Tensor::new(&[2u32, 3, 4], &device)?; 
    let repeat = a.repeat((2))?; 
    // println!("repeat: {}", repeat);
    /* 
        note :: shape(i,j,k) 
                k decides how many times your original tensor will be concatenated.
                all other dims will just repeat the resulting tensor after getting concated k times. 

        repeat => [[[2, 3, 4],
                    [2, 3, 4]],
                   [[2, 3, 4],
                    [2, 3, 4]]]  //[2,2,1]
        returns repeated input tensor (k*j*i) k will be last dim.
    */
    
    let sample = Tensor::rand(0.0, 1.0, (2,4,8), &device)?;
    let transpose = c.transpose(1, 0)?;
    // println!("transpose: {:?}", transpose);
    /* 
        note :: transpose any two dims, narrowed version of permute().
                permute needs all dimensions to be mentioned while transpose needs all necessary 2 dimensions.

        transpose => [4,2,8]
        returns tensor transposed on any two dims 
    */
    
    let sample = Tensor::new(&[[ 0u32,  1 ], [ 4,  5], [ 10,  11] ], &device)?;
    let t = sample.t()?;
    // println!("t: {}", t);
    /* 
        note :: same as using transpose(-1,-2).
                used explicitly when only have to transpose last two dims.
                (python support this operation on 2d vectors only , whereas candle perfoms it on any dims).

        t => [[ 0,  4, 10],
              [ 1,  5, 11]]  // [2,3]
        returns transposed input tensor on last two dimensions.
    */
    
    let on_true = Tensor::full(6u32, (2,4), &device)?;
    let on_false = Tensor::full(3u32,(2,4),&device)?;
    let sample = Tensor::new(&[[7u32, 0, 7, 0],[3, 6, 0, 5]], &device)?;
    let where_cond = sample.where_cond(&on_true, &on_false)?;
    // println!("where_cond: {}", where_cond);
    /* 
        note :: supports only u32,u8,i64 DTypes.

        where_cond => [[6, 3, 6, 3],
                       [6, 6, 3, 6]]  // [2,4]
        returns tensor filled with on_true tensor elements where the input tensor values are non zero
        and on_false tensor elements where the input tensor values are zero.
    */

    // ---------------------------------------------------- End ------------------------------------------------------------\\

    // ------------------------------------------------- Thank You! ---------------------------------------------------------\\

    Ok(())
}
