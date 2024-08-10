use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

const DEVICE: Device = Device::Cpu;

fn main()-> anyhow::Result<()> {

    let file = std::fs::File::open("fetch_california_housing.json")?;
    let reader = std::io::BufReader::new(file);

    // Deserialize the JSON into the struct
    let data: Data = serde_json::from_reader(reader)?;

    // note down the dimensions of the data to reshape the flatten vectors back into original shapes
    let train_d1 = data.X_train.len();
    let train_d2 = data.X_train[0].len();
    let test_d1 = data.X_test.len();
    let test_d2 = data.X_test[0].len();

    // flatten the vectors to make tensors from them 
    // we can not make tensors from multi dimensional vectors
    let x_train_vec = data.X_train.into_iter().flatten().collect::<Vec<_>>();
    let x_test_vec = data.X_test.into_iter().flatten().collect::<Vec<_>>();
    let y_train_vec = data.y_train;
    let y_test_vec = data.y_test;

    // data for training
    let X_train = Tensor::from_vec(x_train_vec.clone(), (train_d1, train_d2), &DEVICE)?;
    let y_train = Tensor::from_vec(y_train_vec, train_d1, &DEVICE)?;

    // data for testing
    let X_test = Tensor::from_vec(x_test_vec.clone(), (test_d1, test_d2), &DEVICE)?;
    let y_test = Tensor::from_vec(y_test_vec, test_d1, &DEVICE)?;

    // make our model to train 

    // varmap is necessary to train/finetune the model as it keeps the track of gradients
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);
    let model = SimpleNN::new(train_d2, vb)?;
    varmap.load("pytorch_model.safetensors")?;

    let optim_config = candle_nn::ParamsAdamW{
        lr: 1e-2,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), optim_config)?;

    train_model(&model, &X_train, &y_train, &mut optimizer, 500)?;
    evaluate_model(&model, &X_test, &y_test)?;

    Ok(())
}

// To load our training data 
#[derive(Debug, serde::Deserialize)]
struct Data{
    X_train: Vec<Vec<f32>>,
    X_test: Vec<Vec<f32>>,
    y_train: Vec<f32>,
    y_test: Vec<f32>,
}

#[derive(Debug)]
struct SimpleNN{
    fc1: Linear,
    fc2: Linear,
}

// This function instantiates a new Model 
impl SimpleNN{
    fn new(in_dim:usize, vb:VarBuilder)->candle_core::Result<Self>{
        let fc1 = linear(in_dim, 64, vb.pp("fc1"))?;
        let fc2 = linear(64, 1, vb.pp("fc2"))?;

        Ok(
            Self { 
                fc1, fc2 
            }
        )
    }
}

// forward pass of our model using Module trait 
impl Module for SimpleNN{
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.fc1.forward(xs)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;

        Ok(x)
    }
}

fn train_model(model:&SimpleNN, X_train:&Tensor, y_train:&Tensor, 
               optimizer:&mut candle_nn::AdamW, epochs:usize)
               ->anyhow::Result<()>{
    for epoch in 0..epochs{
        // forward pass 
        let output = model.forward(X_train)?;
        let loss = candle_nn::loss::mse(&output.squeeze(1)?, y_train)?;

        // backward pass and optimization
        optimizer.backward_step(&loss)?;
        
        if (epoch) % 50 == 0 || epoch == epochs-1{
            println!("Epoch: {}  Train Loss: {}",epoch, loss.to_scalar::<f32>()?);
        }
    }
    Ok(())
}

fn evaluate_model(model:&SimpleNN, X_test:&Tensor, y_test:&Tensor)->anyhow::Result<()>{    
    
    // forward pass 
    let output = model.forward(X_test)?;
    let loss = candle_nn::loss::mse(&output.squeeze(1)?, y_test)?;

    println!("Test Loss: {}", loss.to_scalar::<f32>()?);

    Ok(())
}