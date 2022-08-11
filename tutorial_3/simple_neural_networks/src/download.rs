// script to download and unzip input data 
use std::io::copy;
use std::fs::File;
use std::result::Result;
use std::error::Error;
use tempfile::Builder;

pub async fn download() -> Result<(), Box<dyn Error>> {
    let tmp_dir = Builder::new().prefix("dataset").tempdir()?;
    println!("Downloading file");
    let target = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
    let response = reqwest::get(target).await?;

    let mut dest = {
        let fname = response
            .url()
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|name| if name.is_empty() { None } else { Some(name) })
            .unwrap_or("tmp.bin");

        println!("file to download: '{}'", fname);
        let fname = tmp_dir.path().join(fname);
        println!("will be located under: '{:?}'", fname);
        File::create(fname)?
    };
    let content =  response.text().await?;
    copy(&mut content.as_bytes(), &mut dest)?;
    Ok(())
}