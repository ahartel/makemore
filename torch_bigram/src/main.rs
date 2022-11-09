use std::io::Write;
use std::{collections::HashMap, io};
use tch::Tensor;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

fn read_names() -> Vec<String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(io::stdin());
    rdr.deserialize()
        .map(|r| {
            let s: String = r.unwrap();
            format!(".{s}.").to_string()
        })
        .collect()
}
struct Stoi {
    fwd: HashMap<char, i64>,
    inv: HashMap<i64, char>,
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
        Stoi { fwd, inv }
    }
    pub fn map(&self, c: char) -> i64 {
        self.fwd[&c]
    }
    pub fn inv(&self, idx: i64) -> char {
        self.inv[&idx]
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

fn print_bigrams(bigrams: &Tensor, stoi: &Stoi) {
    let max = bigrams.max();
    for row in 0..27 {
        for col in 0..27 {
            let saturation = bigrams.get(row).get(col) / &max * 256.0;
            let saturation = saturation.int64_value(&[]) as u8;
            print_in_color(
                &format!("  {}{} ", stoi.inv(row), stoi.inv(col)),
                saturation,
            );
        }
        print!("\n");
        for col in 0..27 {
            let saturation = bigrams.get(row).get(col) / &max * 256.0;
            let saturation = saturation.int64_value(&[]) as u8;
            print_in_color(
                &format!("{:0>4} ", bigrams.get(row).get(col).int64_value(&[])),
                saturation,
            );
        }
        print!("\n");
    }
}

fn main() {
    let names = read_names();
    let tokenizer = Stoi::new();
    let bigrams = extract_bigrams(&names, &tokenizer);
    print_bigrams(&bigrams, &tokenizer);
    let smoothing = bigrams.ones_like();
    let row_sums = (&smoothing + &bigrams).sum_to_size(&[27, 1]);
    let probs = (bigrams + smoothing) / row_sums;
    sample_words_bigram(5, &probs, &tokenizer);
    print_log_likelihood_bigram(&names, &probs, &tokenizer);
}

fn print_log_likelihood_bigram(names: &[String], probs: &Tensor, stoi: &Stoi) {
    let mut total_log_likelihood: f64 = 0.0;
    let mut n = 0;
    for name in names {
        for (c1, c2) in name.chars().zip(name[1..].chars()) {
            let idx1 = stoi.map(c1);
            let idx2 = stoi.map(c2);
            n += 1;
            total_log_likelihood += probs.get(idx1).get(idx2).log().double_value(&[]);
        }
    }
    println!("{}", -total_log_likelihood / n as f64);
}

fn sample_words_bigram(num: usize, probs: &Tensor, stoi: &Stoi) {
    for _ in 0..num {
        let mut chars = Vec::new();
        let mut sampled = 0;
        loop {
            sampled = probs.get(sampled).multinomial(1, true).int64_value(&[]);
            chars.push(stoi.inv(sampled));
            if sampled == 0 {
                break;
            }
        }
        let name: String = chars.into_iter().collect();
        println!("{:?}", name);
    }
}

fn extract_bigrams(names: &[String], stoi: &Stoi) -> Tensor {
    let mut bigrams = [[0; 27]; 27];
    for word in names {
        for (c1, c2) in word.chars().zip(word[1..].chars()) {
            let idx1 = stoi.map(c1) as usize;
            let idx2 = stoi.map(c2) as usize;
            bigrams[idx1][idx2] += 1;
        }
    }
    Tensor::of_slice2(&bigrams)
}
