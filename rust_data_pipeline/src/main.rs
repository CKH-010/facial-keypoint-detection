/* 
Technical Summary:
We built a cross-language pipeline (Python to Rust) to analyze the feature maps extracted from NaimishNet using forward hooks. 
We applied Welch's T-test to compare the activation distributions of the early layers vs. deep layers.

The Data:
Early Layer (Layer 1): [1, 32, 93, 93] (High spatial resolution)
Deep Layer (Layer 4): [1, 256, 10, 10] (Severe spatial compression)
T-statistic: -117.52

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
    let array_layer4: ArrayD<f32> = read_npy("layer4_features.npy").unwrap();
    println!("Layer 4 Shape: {:?}", array_layer4.shape());

    let array_layer1: ArrayD<f32> = read_npy("layer1_features.npy").unwrap();
    println!("Layer 1 Shape: {:?}", array_layer1.shape());

    println!("Layer 4 Std Dev: {:.6}", array_layer4.std(0.0));  
    println!("Layer 1 Std Dev: {:.6}", array_layer1.std(0.0));

    let flattened_layer4 = array_layer4
        .clone()
        .into_shape_with_order(array_layer4.len())
        .unwrap();
    let flattened_layer1 = array_layer1
        .clone()
        .into_shape_with_order(array_layer1.len())
        .unwrap();

    let t_stat = welch_t_statistic(&flattened_layer4.to_vec(), &flattened_layer1.to_vec());
    println!("T-statistic: {:?}", t_stat);
}
