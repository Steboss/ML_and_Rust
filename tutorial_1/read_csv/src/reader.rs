use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use std::vec::Vec; 

#[derive(Debug)]  // attribute on the following struct to auto-generate suitable implementation of the debug trait
pub struct HousingDataset{
    // Define a struct for the housing dataset
    // to find feature reference:
    // https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
    crim: f64,
    zn: f64,
    indus: f64,
    chas: f64,
    nox: f64,
    rm: f64,
    age: f64,
    dis: f64,
    rad: f64,
    tax: f64,
    ptratio: f64, 
    black: f64,
    lstat: f64, 
    medv: f64,
}
// To the struct above we are going to IMPLement a functionality. Namely we are going 
// to implement a reading functionality
impl HousingDataset{
    pub fn new(v: Vec<&str>) -> HousingDataset {
        // note we have declared a new vector of &strings, remember borrowship, which will be a struct HousingDatset
        // as input we are receiving a string from the csv file. 
        // we are going to decode this as a f64 vector using the unwrap function 
        let unwrapped_text: Vec<f64> = v.iter().map(|r| r.parse().unwrap()).collect(); 
        // the input string v is iterate, so each row r is parsed and unwrap
        HousingDataset{crim: unwrapped_text[0], 
                        zn: unwrapped_text[1], 
                        indus: unwrapped_text[2],
                        chas: unwrapped_text[3],
                        nox: unwrapped_text[4],
                        rm: unwrapped_text[5],
                        age: unwrapped_text[6],
                        dis: unwrapped_text[7],
                        rad: unwrapped_text[8],
                        tax: unwrapped_text[9],
                        ptratio: unwrapped_text[10],
                        black: unwrapped_text[11],
                        lstat: unwrapped_text[12],
                        medv: unwrapped_text[13]} // notice here no ; as we're returning this
    }

    pub fn train_features(&self) -> Vec<f64> {
        // return all the features to be used in train 
        // vec! define a simple vector, like a list in python 
        vec![self.crim, 
            self.zn,
            self.indus, 
            self.chas, 
            self.nox, 
            self.rm, 
            self.age, 
            self.dis, 
            self.rad, 
            self.tax, 
            self.ptratio, 
            self.black, 
            self.lstat] // remember we're returning this so no ; 
    }

    pub fn train_target(&self) -> f64 {
        // this is just a number, namely the price 
        self.medv
    }
}

fn read_single_line(s: String) -> HousingDataset { 
    // read a single line 
    let raw_vector: Vec<&str> = s.split_whitespace().collect(); 
    // now read the single vector
    let housing_vector: HousingDataset = HousingDataset::new(raw_vector); 
    // now return the vector, no ; 
    housing_vector
}
// finally let's define a function to read the housing input file 
pub fn read_housing_csv(filename: impl AsRef<Path>) -> Vec<HousingDataset> {
    let file = File::open(filename).expect("Please, give an input file. Cannot find file");
    // notice how to read a file in rust. The .expect is used in case of error 
    let reader = BufReader::new(file);
    // from Rust: 
    /*It can be excessively inefficient to work directly with a Read instance. 
    For example, every call to read on TcpStream results in a system call. 
    A BufReader<R> performs large, infrequent reads on the underlying 
    Read and maintains an in-memory buffer of the results.
     */
    reader.lines().enumerate()
            .map(|(numb, line)| line.expect(&format!("Impossible to read line number {}", numb)))
            .map(|row| read_single_line(row))
            .collect()
}