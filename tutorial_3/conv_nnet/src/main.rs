// simple script to run a linear neural network against input images 
//mod download;

use std::result::Result;
use std::error::Error;
use mnist::*;
use tch::{kind, Kind, Tensor, nn, nn::ModuleT, nn::OptimizerConfig, Device};
use ndarray::{Array3, Array2};


const LABELS: i64 = 10; // number of distinct labels
const HEIGHT: usize = 28; 
const WIDTH: usize = 28;

const TRAIN_SIZE: usize = 1000;
const VAL_SIZE: usize = 200;
const TEST_SIZE: usize =200;

const N_EPOCHS: i64 = 200;

const THRES: f64 = 0.001;

const BATCH_SIZE: i64 = 256;


#[derive(Debug)]
struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        // stride -- padding -- dilation
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 1024, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, 10, Default::default());
        Net { conv1, conv2, fc1, fc2 }
    }
}

// forward step
impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply(&self.fc1)
            .relu()
            .dropout(0.5, train)
            .apply(&self.fc2)
    }
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
    let mut ws = Tensor::zeros(&[(HEIGHT*WIDTH) as i64, LABELS], kind::FLOAT_CPU).set_requires_grad(true);
    let mut bs = Tensor::zeros(&[LABELS], kind::FLOAT_CPU).set_requires_grad(true);
    let train_data = image_to_tensor(trn_img, TRAIN_SIZE, HEIGHT, WIDTH);
    let train_lbl = labels_to_tensor(trn_lbl, TRAIN_SIZE, 1);
    let test_data = image_to_tensor(tst_img, TEST_SIZE, HEIGHT, WIDTH); 
    let test_lbl = labels_to_tensor(tst_lbl, TEST_SIZE, 1);
    let val_data = image_to_tensor(val_img, VAL_SIZE, HEIGHT, WIDTH);
    let val_lbl = labels_to_tensor(val_lbl, VAL_SIZE, 1);

    // set up variable store to check if cuda is available 
    let vs = nn::VarStore::new(Device::cuda_if_available());
    // set up the Conv net 
    let net = Net::new(&vs.root());
    // set up optimizer 
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    let n_it = (TRAIN_SIZE as i64)/BATCH_SIZE; // already ceiled
    println!("Number of iteration with given batch size: {:?}", n_it);
    // run epochs 
    for epoch in 1..100 {
        // generate random idxs for batch size 
        // run all the images divided in batches  -> for loop
        for i in 1..n_it {
            let batch_idxs = generate_random_index(TRAIN_SIZE as i64, BATCH_SIZE); 
            let batch_images = train_data.index_select(0, &batch_idxs).to_device(vs.device()).to_kind(Kind::Float); 
            let batch_lbls = train_lbl.index_select(0, &batch_idxs).to_device(vs.device()).to_kind(Kind::Int64);
            // compute the loss 
            let loss = net.forward_t(&batch_images, true).cross_entropy_for_logits(&batch_lbls);
            opt.backward_step(&loss);
        }
        // compute accuracy 
        let test_accuracy =
            net.batch_accuracy_for_logits(&test_data, &test_lbl, vs.device(), 1024);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }

    Ok(())
}
