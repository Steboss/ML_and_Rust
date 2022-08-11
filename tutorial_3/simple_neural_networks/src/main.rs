// simple script to run a linear neural network against input images 
//mod download;

use std::result::Result;
use std::error::Error;
use mnist::*;
use tch::{kind, no_grad, Kind, Tensor};
use ndarray::{Array3, Array2};


const LABELS: i64 = 10; // number of distinct labels
const HEIGHT: usize = 28; 
const WIDTH: usize = 28;

const TRAIN_SIZE: usize = 1000;
const VAL_SIZE: usize = 200;
const TEST_SIZE: usize =200;

const N_EPOCHS: i64 = 200;

const THRES: f64 = 0.001;


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
    let mut ws = Tensor::zeros(&[(HEIGHT*WIDTH) as i64, LABELS], kind::FLOAT_CPU).set_requires_grad(true);
    let mut bs = Tensor::zeros(&[LABELS], kind::FLOAT_CPU).set_requires_grad(true);
    let train_data = image_to_tensor(trn_img, TRAIN_SIZE, HEIGHT, WIDTH);
    let train_lbl = labels_to_tensor(trn_lbl, TRAIN_SIZE, 1);
    let test_data = image_to_tensor(tst_img, TEST_SIZE, HEIGHT, WIDTH); 
    let test_lbl = labels_to_tensor(tst_lbl, TEST_SIZE, 1);
    let val_data = image_to_tensor(val_img, VAL_SIZE, HEIGHT, WIDTH);
    let val_lbl = labels_to_tensor(val_lbl, VAL_SIZE, 1);


    // run epochs 
    let mut loss_diff;
    let mut curr_loss = 0.0;

    'train: for epoch in 1..N_EPOCHS{
        // neural network multiplication
        let logits = train_data.matmul(&ws) + &bs; 
        // compute the loss as log softmax
        let loss = logits.log_softmax(-1, Kind::Float).nll_loss(&train_lbl);
        // gradient
        ws.zero_grad();
        bs.zero_grad();
        loss.backward();
        // back propgation
        no_grad(|| {
            ws += ws.grad()*(-1);
            bs += bs.grad()*(-1);
        });
        // validation
        let val_logits = val_data.matmul(&ws) + &bs;
        let val_accuracy = val_logits
            .argmax(Some(-1), false)
            .eq_tensor(&val_lbl)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);

        println!(
            "epoch: {:4} train loss: {:8.5} val acc: {:5.2}%",
            epoch,
            loss.double_value(&[]),
            100. * val_accuracy
        );
        // early stop 
        if epoch == 1{
            curr_loss = loss.double_value(&[]);
        } else {
            loss_diff = (loss.double_value(&[]) - curr_loss).abs(); 
            curr_loss = loss.double_value(&[]); 
            // if we are less then threshold stop 
            if loss_diff < THRES {
                println!("Target accuracy reached, early stopping");
                break 'train;
            }
        }

    } 

    // the final weight and bias gives us the test accuracy
    let test_logits = test_data.matmul(&ws) + &bs; 
    let test_accuracy = test_logits
        .argmax(Some(-1), false)
        .eq_tensor(&test_lbl)
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[]);
    println!("Final test accuracy {:5.2}%", 100.*test_accuracy);

    Ok(())
}
