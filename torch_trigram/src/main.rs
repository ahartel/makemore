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
            format!("..{s}.").to_string()
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
    let stoi = Stoi::new();
    let trigrams = extract_trigrams(&names, &stoi);
    // print_bigrams(&trigrams, &stoi);
    let smoothing = trigrams.ones_like();
    let row_sums = (&smoothing + &trigrams).sum_to_size(&[27 * 27, 1]);
    let probs = (trigrams + smoothing) / row_sums;
    sample_words(5, &probs, &stoi);
    print_log_likelihood(&names, &probs, &stoi);
}

fn print_log_likelihood(names: &[String], probs: &Tensor, stoi: &Stoi) {
    let mut total_log_likelihood: f64 = 0.0;
    let mut n = 0;
    for name in names {
        for ((c1, c2), c3) in name.chars().zip(name[1..].chars()).zip(name[2..].chars()) {
            let idx1 = stoi.map(c1);
            let idx2 = stoi.map(c2);
            let idx3 = stoi.map(c3);
            n += 1;
            total_log_likelihood += probs
                .get(idx1 * 27 + idx2)
                .get(idx3)
                .log()
                .double_value(&[]);
        }
    }
    println!("{}", -total_log_likelihood / n as f64);
}

fn sample_words(num: usize, probs: &Tensor, stoi: &Stoi) {
    for _ in 0..num {
        let mut chars = Vec::new();
        let mut last_sampled = 0;
        let mut prev_sampled = 0;
        loop {
            let new_sampled = probs
                .get(prev_sampled * 27 + last_sampled)
                .multinomial(1, true)
                .int64_value(&[]);
            chars.push(stoi.inv(new_sampled));
            if new_sampled == 0 {
                break;
            }
            prev_sampled = last_sampled;
            last_sampled = new_sampled;
        }
        let name: String = chars.into_iter().collect();
        println!("{:?}", name);
    }
}

fn extract_trigrams(names: &[String], stoi: &Stoi) -> Tensor {
    let mut trigrams: Vec<Vec<i64>> = vec![vec![0; 27]; 27 * 27];
    for word in names {
        for ((c1, c2), c3) in word.chars().zip(word[1..].chars()).zip(word[2..].chars()) {
            let idx1 = stoi.map(c1) as usize;
            let idx2 = stoi.map(c2) as usize;
            let idx3 = stoi.map(c3) as usize;
            trigrams[idx1 * 27 + idx2][idx3] += 1;
        }
    }
    Tensor::of_slice2(&trigrams)
}
