// simple script to run a linear neural network against input images 
//mod download;

use std::result::Result;
use std::error::Error;
use mnist::*;
use tch::{kind, Kind, Tensor, nn, nn::Module, nn::OptimizerConfig, Device};
use ndarray::{Array3, Array2};


const LABELS: i64 = 10; // number of distinct labels
const HEIGHT: usize = 28; 
const WIDTH: usize = 28;
const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;

const TRAIN_SIZE: usize = 50000;
const VAL_SIZE: usize = 10000;
const TEST_SIZE: usize =10000;

const N_EPOCHS: i64 = 200;

const THRES: f64 = 0.001;

const BATCH_SIZE: i64 = 256;

fn net(vs: &nn::Path) -> impl Module{
    nn::seq()
    .add(nn::linear(vs/"layer1", IMAGE_DIM, HIDDEN_NODES, Default::default() ))
    .add_fn(|xs| xs.relu())
    .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}


pub fn image_to_tensor(data:Vec<u8>, dim1:usize, dim2:usize, dim3:usize)-> Tensor{
    // normalize the image as well 
    let inp_data: Array3<f32> = Array3::from_shape_vec((dim1, dim2, dim3), data)
        .expect("Error converting data to 3D array")
        .map(|x| *x as f32/256.0);
    // convert to tensor
    let inp_tensor = Tensor::of_slice(inp_data.as_slice().unwrap());
    // reshape so we'll have dim1, dim2*dim3 shape array
    let ax1 = dim1 as i64; 
    let ax2 = (dim2 as i64)*(dim3 as i64);
    let shape: Vec<i64>  = vec![ ax1, ax2 ];
    let output_data = inp_tensor.reshape(&shape);
    println!("Output image tensor size {:?}", shape);
        
    output_data
}


pub fn labels_to_tensor(data:Vec<u8>, dim1:usize, dim2:usize)-> Tensor{
    let inp_data: Array2<i64> = Array2::from_shape_vec((dim1, dim2), data)
        .expect("Error converting data to 2D array")
        .map(|x| *x as i64);

    let output_data = Tensor::of_slice(inp_data.as_slice().unwrap());
    println!("Output label tensor size {:?}", output_data.size());
    
    output_data
}


pub fn generate_random_index(ArraySize: i64, BatchSize: i64)-> Tensor{
    let random_idxs = Tensor::randint(ArraySize, &[BatchSize], kind::INT64_CPU);
    random_idxs
}


fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main()-> Result<(), Box<dyn Error>> { 

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        val_img, 
        val_lbl,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_digit()
        .training_set_length(TRAIN_SIZE as u32)
        .validation_set_length(VAL_SIZE as u32)
        .test_set_length(TEST_SIZE as u32)
        .finalize();
    
    // set up a weight and bias tensor
    let train_data = image_to_tensor(trn_img, TRAIN_SIZE, HEIGHT, WIDTH);
    let train_lbl = labels_to_tensor(trn_lbl, TRAIN_SIZE, 1);
    let test_data = image_to_tensor(tst_img, TEST_SIZE, HEIGHT, WIDTH); 
    let test_lbl = labels_to_tensor(tst_lbl, TEST_SIZE, 1);
    let val_data = image_to_tensor(val_img, VAL_SIZE, HEIGHT, WIDTH);
    let val_lbl = labels_to_tensor(val_lbl, VAL_SIZE, 1);

    // set up variable store to check if cuda is available 
    let vs = nn::VarStore::new(Device::cuda_if_available());
    // set up the seq net 
    let net = net(&vs.root());
    // set up optimizer 
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    println!("Number of iteration with given batch size: {:?}", n_it);
    // run epochs 
    for epoch in 1..N_EPOCHS {
        let loss = net.forward(&train_data).cross_entropy_for_logits(&train_lbl);
        // backward step 
        opt.backward_step(&loss);
        //accuracy on test
        let val_accuracy = net.forward(&val_data).accuracy_for_logits(&val_lbl);
        println!(
            "epoch: {:4} train loss: {:8.5} val acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&val_accuracy),
        );
    }

    // final test 
    let test_accuracy = net.forward(&test_data).accuracy_for_logits(&test_lbl);
    println!("Final test accuracy {:5.2}%", 100.*f64::from(&test_accuracy));

    Ok(())
}
