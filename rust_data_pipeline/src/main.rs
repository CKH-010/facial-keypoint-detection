/* 
Technical Summary:
We built a cross-language pipeline (Python to Rust) to analyze the feature maps extracted from NaimishNet using forward hooks. 
We applied Welch's T-test to compare the activation distributions of the early layers vs. deep layers.

The Data:

Baseline: L1 SD 0.179 → L4 SD 0.0375 (5× drops), T +56.5 (L4 mean higher than L1). Deep layer collapsed — tiny variance, everything saturates to a similar value.
Advanced: L1 SD 0.609 → L4 SD 0.255 (only ~2.4× drop, and stays 6.8× larger than baseline's L4), T −59.96 (L4 mean lower than L1). Deep layer stays expressive, distributed, doesn't collapse.

Baseline's deep layer: high mean, tiny spread (collapsed/saturated).
Advanced's deep layer: lower mean, wide spread (still representing features).

It further demonstrates the significant difference in activation distributions between early and deep layers, as indicated by the large negative T-statistic.

** Conclusion: The large negative T-statistic indicates that the early and deep layers have significantly different activation distributions. **

Ps: we are using rust for this is to test the cross-language pipeline and performance of Rust in handling large feature maps.

*/

use ndarray::ArrayD;
use ndarray_npy::read_npy;

fn welch_t_statistic(sample_a: &[f32], sample_b: &[f32]) -> Option<f64> {
    if sample_a.len() < 2 || sample_b.len() < 2 {
        return None;
    }

    let mean_a = sample_a.iter().map(|&x| x as f64).sum::<f64>() / sample_a.len() as f64;
    let mean_b = sample_b.iter().map(|&x| x as f64).sum::<f64>() / sample_b.len() as f64;
    
    let variance_a = sample_a
        .iter()
        .map(|&x| (x as f64 - mean_a).powi(2))
        .sum::<f64>()
        / (sample_a.len() - 1) as f64;
    
    let variance_b = sample_b
        .iter()
        .map(|&x| (x as f64 - mean_b).powi(2))
        .sum::<f64>()
        / (sample_b.len() - 1) as f64;
    
    let standard_error =
        (variance_a / sample_a.len() as f64 + variance_b / sample_b.len() as f64).sqrt();

    (standard_error > 0.0).then_some((mean_a - mean_b) / standard_error)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: cargo run -- <layer1.npy> <layer4.npy>");
        std::process::exit(1);
    }

    let layer1: ArrayD<f32> = read_npy(&args[1]).expect("Cannot read layer 1");
    let layer4: ArrayD<f32> = read_npy(&args[2]).expect("Cannot read layer 4");

    println!("Layer 1 shape: {:?}, SD: {:.6}", layer1.shape(), layer1.std(0.0));
    println!("Layer 4 shape: {:?}, SD: {:.6}", layer4.shape(), layer4.std(0.0));

    let values1: Vec<f32> = layer1.iter().copied().collect();
    let values4: Vec<f32> = layer4.iter().copied().collect();

    println!("Layer 4 minus layer 1 T-statistic: {:?}",
        welch_t_statistic(&values4, &values1));
}
