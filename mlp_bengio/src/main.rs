use itertools::izip;
use std::borrow::Cow;
use std::io::Write;
use std::{collections::HashMap, io};
use tch::{Device, Kind, Tensor};
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

const DEVICE: Device = Device::Cpu;

fn read_names() -> Vec<String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(io::stdin());
    rdr.deserialize()
        .map(|r| {
            let s: String = r.unwrap();
            format!("...{s}.").to_string()
        })
        .collect()
}

#[derive(Clone)]
struct Stoi {
    _fwd: HashMap<char, i64>,
    _inv: HashMap<i64, char>,
}

impl Stoi {
    pub fn new() -> Self {
        let fwd = ".abcdefghijklmnopqrstuvwxyz"
            .chars()
            .zip(0..27)
            .collect::<HashMap<char, i64>>();
        let inv = (0..27)
            .zip(".abcdefghijklmnopqrstuvwxyz".chars())
            .collect::<HashMap<i64, char>>();
        Stoi {
            _fwd: fwd,
            _inv: inv,
        }
    }
    pub fn map(&self, c: char) -> i64 {
        self._fwd[&c]
    }
    pub fn inv(&self, idx: i64) -> char {
        self._inv[&idx]
    }
}

fn print_in_color(s: &str, saturation: u8) {
    let mut stdout = StandardStream::stdout(ColorChoice::Always);
    stdout
        .set_color(ColorSpec::new().set_fg(Some(Color::Rgb(saturation, saturation, saturation))))
        .unwrap();
    write!(&mut stdout, "{s}").unwrap();
    stdout.set_color(ColorSpec::new().set_fg(None)).unwrap();
}

#[derive(Debug)]
struct Error(Cow<'static, str>);

/// print a 2D tensor who's values will be interpreted as doubles
/// xlables and ylabels must be provided
/// relative magnitude of values will be indicated by grey values
fn print_tensor_2d(
    tensor: &Tensor,
    xlabels: impl Fn(i64) -> String,
    ylabels: impl Fn(i64) -> String,
) -> Result<(), Error> {
    if tensor.size().len() != 2 {
        return Err(Error("Tensor must have 2 dimensions".into()));
    }
    let max = tensor.max();
    let min = tensor.min();
    let size = tensor.size();
    for row in 0..size[0] {
        for col in 0..size[1] {
            let value = tensor.get(row).get(col);
            let saturation = (value - &min) / (&max - &min) * 256.0;
            let saturation = saturation.int64_value(&[]) as u8;
            print_in_color(&format!("{}{}    ", ylabels(row), xlabels(col)), saturation);
        }
        print!("\n");
        for col in 0..size[1] {
            let value = tensor.get(row).get(col);
            let saturation = (&value - &min) / (&max - &min) * 256.0;
            let saturation = saturation.int64_value(&[]) as u8;
            print_in_color(&format!("{:+.2} ", value.double_value(&[])), saturation);
        }
        print!("\n");
    }
    Ok(())
}

fn prepare_labels(stoi: &Stoi) -> (Box<dyn Fn(i64) -> String>, Box<dyn Fn(i64) -> String>) {
    let stoi_x = stoi.clone();
    let stoi_y = stoi.clone();
    let xlabels = Box::new(move |x| String::from(stoi_x.inv(x)));
    let ylabels = Box::new(move |y| String::from(stoi_y.inv(y)));
    (xlabels, ylabels)
}

fn forward_pass(training_x: &Tensor, parameters: &[Tensor; 5]) -> Tensor {
    let emb = parameters[0].index(&[Some(training_x)]);
    let emb = emb.view([emb.size()[0], emb.size()[1] * emb.size()[2]]);
    let hidden = (emb.matmul(&parameters[1]) + &parameters[2]).tanh();
    let logits = hidden.matmul(&parameters[3]) + &parameters[4];
    logits
}

fn prepare_training_dataset(names: &[String], tokenizer: &Stoi) -> (Tensor, Tensor) {
    const BLOCK_SIZE: usize = 3;
    let mut xs: Vec<[i64; BLOCK_SIZE]> = Vec::new();
    let mut ys: Vec<i64> = Vec::new();
    for name in names.iter().take(10) {
        for (x1, x2, x3, y) in izip!(
            name.chars(),
            name.chars().skip(1),
            name.chars().skip(2),
            name.chars().skip(3)
        ) {
            let idx_1 = tokenizer.map(x1);
            let idx_2 = tokenizer.map(x2);
            let idx_3 = tokenizer.map(x3);
            xs.push([idx_1, idx_2, idx_3]);
            let idx_y = tokenizer.map(y);
            ys.push(idx_y);
        }
    }
    let xs = Tensor::of_slice2(&xs);
    let ys = Tensor::of_slice(&ys);
    (xs, ys)
}

fn main() {
    let names = read_names();
    println!("Loaded {} names", names.len());
    for name in names.iter().take(8) {
        println!("{}", name);
    }
    let tokenizer = Stoi::new();
    let mut parameters: [Tensor; 5] = [
        Tensor::randn(&[27, 2], (Kind::Float, DEVICE)).set_requires_grad(true),
        Tensor::randn(&[6, 100], (Kind::Float, DEVICE)).set_requires_grad(true),
        Tensor::randn(&[100], (Kind::Float, DEVICE)).set_requires_grad(true),
        Tensor::randn(&[100, 27], (Kind::Float, DEVICE)).set_requires_grad(true),
        Tensor::randn(&[27], (Kind::Float, DEVICE)).set_requires_grad(true),
    ];
    let (training_x, training_y) = prepare_training_dataset(&names, &tokenizer);

    for _ in 0..100 {
        let logits = forward_pass(&training_x, &parameters);
        let loss = logits.cross_entropy_for_logits(&training_y);
        // let probs = logits.softmax(1, Kind::Float);
        // let loss = -probs
        //     .index(&[
        //         Some(Tensor::arange(probs.size()[0], (Kind::Int64, DEVICE))),
        //         Some(training_y),
        //     ])
        //     .log()
        //     .mean(Kind::Float)
        //     .double_value(&[]);
        println!("Loss: {}", loss.double_value(&[]));
        backward_pass(&loss, &mut parameters);
    }
    let logits = forward_pass(&training_x, &parameters);
    let probs = logits.softmax(1, Kind::Float);
    let (xlabels, ylabels) = prepare_labels(&tokenizer);
    print_tensor_2d(&probs, xlabels, ylabels).unwrap();
}

fn backward_pass(loss: &Tensor, parameters: &mut [Tensor; 5]) {
    let rate = -0.1;

    for param in parameters.iter_mut() {
        param.zero_grad();
    }

    loss.backward();

    for param in parameters {
        param.set_data(&(param.data() + rate * param.grad()));
    }
}
