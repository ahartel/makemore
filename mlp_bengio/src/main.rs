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

fn prepare_labels(stoi: &Stoi) -> (impl Fn(i64) -> String, impl Fn(i64) -> String) {
    let stoi_x = stoi.clone();
    let stoi_y = stoi.clone();
    let xlabels = move |x| String::from(stoi_x.inv(x));
    let ylabels = move |y| String::from(stoi_y.inv(y));
    (xlabels, ylabels)
}

fn forward_pass(training_x: &Tensor, parameters: &[Tensor; 5]) -> Tensor {
    let emb = parameters[0].index(&[Some(training_x)]);
    let emb = emb.view([emb.size()[0], emb.size()[1] * emb.size()[2]]);
    let hidden = (emb.matmul(&parameters[1]) + &parameters[2]).tanh();
    let logits = hidden.matmul(&parameters[3]) + &parameters[4];
    logits
}

fn prepare_dataset(
    names: &[String],
    tokenizer: &Stoi,
    skip: usize,
    take: usize,
) -> (Tensor, Tensor) {
    const BLOCK_SIZE: usize = 3;
    let mut xs: Vec<[i64; BLOCK_SIZE]> = Vec::new();
    let mut ys: Vec<i64> = Vec::new();
    for name in names.iter().skip(skip).take(take) {
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

fn backward_pass(loss: &Tensor, parameters: &mut [Tensor; 5], learning_rate: f32) {
    for param in parameters.iter_mut() {
        param.zero_grad();
    }

    loss.backward();

    for param in parameters {
        param.set_data(&(param.data() + learning_rate * param.grad()));
    }
}

fn sample_words(num: usize, parameters: &[Tensor; 5], stoi: &Stoi) {
    for _ in 0..num {
        let mut chars = Vec::new();
        let mut input = Tensor::zeros(&[1, 3], (Kind::Int64, DEVICE));
        loop {
            let next_logits = forward_pass(&input, parameters);
            let probs = next_logits.softmax(1, Kind::Float);
            let sampled = probs.multinomial(1, true);
            let sampled_int_value = sampled.int64_value(&[]);
            chars.push(stoi.inv(sampled_int_value));
            if sampled_int_value == 0 {
                break;
            }
            input = Tensor::cat(&[input.slice(1, 1, 3, 1), sampled], 1);
        }
        let name: String = chars.into_iter().collect();
        println!("{:?}", name);
    }
}

fn main() {
    let names = read_names();
    println!("Loaded {} names", names.len());
    for name in names.iter().take(8) {
        println!("{}", name);
    }
    let tokenizer = Stoi::new();
    let mut parameters: [Tensor; 5] = [
        // embedding
        Tensor::randn(&[27, 5], (Kind::Float, DEVICE)).set_requires_grad(true),
        // hidden layer weights and biases
        Tensor::randn(&[15, 50], (Kind::Float, DEVICE)).set_requires_grad(true),
        Tensor::randn(&[50], (Kind::Float, DEVICE)).set_requires_grad(true),
        // output layer weights and biases
        Tensor::randn(&[50, 27], (Kind::Float, DEVICE)).set_requires_grad(true),
        Tensor::randn(&[27], (Kind::Float, DEVICE)).set_requires_grad(true),
    ];
    let training_end_index = names.len() / 100 * 80;
    let (training_x, training_y) = prepare_dataset(&names, &tokenizer, 0, training_end_index);
    let dev_end_index = training_end_index + names.len() / 100 * 10;
    let (dev_x, dev_y) = prepare_dataset(
        &names,
        &tokenizer,
        training_end_index,
        names.len() / 100 * 10,
    );
    let (test_x, test_y) =
        prepare_dataset(&names, &tokenizer, dev_end_index, names.len() / 100 * 10);
    println!("{:?},{:?}", training_x.size(), training_y.size());
    for run in 0..100000 {
        let batch_indices = Tensor::randint(training_x.size()[0], &[32], (Kind::Int64, DEVICE));
        let logits = forward_pass(&training_x.index(&[Some(&batch_indices)]), &parameters);
        let loss = logits.cross_entropy_for_logits(&training_y.index(&[Some(&batch_indices)]));
        let learning_rate = if run < 50000 { -0.1 } else { -0.01 };
        backward_pass(&loss, &mut parameters, learning_rate);
    }
    let logits = forward_pass(&training_x, &parameters);
    let loss = logits.cross_entropy_for_logits(&training_y);
    println!("Loss on train: {}", loss.double_value(&[]));

    let logits = forward_pass(&dev_x, &parameters);
    let loss = logits.cross_entropy_for_logits(&dev_y);
    println!("Loss on dev: {}", loss.double_value(&[]));
    sample_words(10, &parameters, &tokenizer);
    // let probs = logits.softmax(1, Kind::Float);
    // let (xlabels, ylabels) = prepare_labels(&tokenizer);
    // print_tensor_2d(&probs, xlabels, ylabels).unwrap();
}
