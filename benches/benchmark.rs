use count_measurement::CountMeasurement;
use criterion::{
    BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::Measurement,
};
use history::*;
use std::{convert::Infallible, hint::black_box, marker::PhantomData};

#[derive(Default, Clone)]
struct Noop<State> {
    _state: PhantomData<State>,
}
impl<State: Clone> Action for Noop<State> {
    type State = State;

    fn apply(&self, state: Self::State, _: &mut Self::Context) -> Result<Self::State, Self::Error> {
        Ok(state)
    }
}

#[derive(Default, Clone)]
struct CountApplyAction;
impl Action for CountApplyAction {
    type State = u8;

    fn apply(&self, state: Self::State, _: &mut Self::Context) -> Result<Self::State, Self::Error> {
        CountMeasurement::increment();
        Ok(state)
    }
}

#[derive(Default)]
struct CloneCountState(u8);
impl Clone for CloneCountState {
    fn clone(&self) -> Self {
        CountMeasurement::increment();
        Self(self.0)
    }
}

trait BenchmarkConfig {
    type Action: Default + Clone + Action<State: Default, Context = (), Error = Infallible>;
    fn name() -> &'static str;
}

struct TimingConfig;
impl BenchmarkConfig for TimingConfig {
    type Action = Noop<u8>;
    fn name() -> &'static str {
        "timing"
    }
}

struct CountApplyConfig;
impl BenchmarkConfig for CountApplyConfig {
    type Action = CountApplyAction;
    fn name() -> &'static str {
        "count_apply"
    }
}

struct CountCloneConfig;
impl BenchmarkConfig for CountCloneConfig {
    type Action = Noop<CloneCountState>;
    fn name() -> &'static str {
        "count_clone"
    }
}

fn filled_history<Config: BenchmarkConfig>(count: u64) -> History<Config::Action> {
    let mut history = History::default();
    for _ in 0..count {
        black_box(history.push_action(Config::Action::default()));
    }
    history
}

fn criterion_benchmark<Config: BenchmarkConfig>(c: &mut Criterion<impl Measurement>) {
    let counts = [10, 100, 1000, 10000];
    {
        let mut group = c.benchmark_group(format!("{}/push_action", Config::name()));
        for &count in &counts {
            group.throughput(criterion::Throughput::Elements(count));
            group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
                b.iter_batched_ref(
                    History::default,
                    |h| {
                        for _ in 0..count {
                            black_box(h.push_action(Config::Action::default()));
                        }
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    {
        let mut group = c.benchmark_group(format!("{}/pop_action", Config::name()));
        for &count in &counts {
            group.throughput(criterion::Throughput::Elements(count));
            group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
                let h = filled_history::<Config>(count);
                b.iter_batched_ref(
                    || h.clone(),
                    |h| {
                        for _ in 0..count {
                            black_box(h.pop_action());
                        }
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    {
        // Removing every action, to compare against `pop_action` above at the same sizes.
        let mut group = c.benchmark_group(format!("{}/pop_actions/all", Config::name()));
        for &count in &counts {
            group.throughput(criterion::Throughput::Elements(count));
            group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
                let h = filled_history::<Config>(count);
                b.iter_batched_ref(
                    || h.clone(),
                    |h| {
                        black_box(h.pop_actions(count as usize));
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    {
        // Removing only the most recent `k` of a fixed number of actions, where the *O*(*k* (1 +
        // log (*n*/*k*))) cost of a single call is expected to beat `k` separate pops. Removing
        // all of them is degenerate, because no cached state has to be recomputed, so every `k`
        // here is strictly less than `n`.
        let count = *counts.last().unwrap();
        let mut group = c.benchmark_group(format!("{}/pop_actions/n={count}", Config::name()));
        let h = filled_history::<Config>(count);
        for k in [1, 10, 100, 1000] {
            group.throughput(criterion::Throughput::Elements(k));
            group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
                b.iter_batched_ref(
                    || h.clone(),
                    |h| {
                        black_box(h.pop_actions(k as usize));
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
}

fn count_criterion_config() -> Criterion<impl Measurement> {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_nanos(1))
        .with_measurement(CountMeasurement)
}

criterion_group!(time_benches, criterion_benchmark<TimingConfig>);

criterion_group! {
    name = apply_count_benches;
    config = count_criterion_config();
    targets = criterion_benchmark<CountApplyConfig>,
}

criterion_group! {
    name = clone_count_benches;
    config = count_criterion_config();
    targets = criterion_benchmark<CountCloneConfig>,
}

criterion_main!(apply_count_benches, clone_count_benches, time_benches);

mod count_measurement {
    use criterion::{
        Throughput,
        measurement::{Measurement, ValueFormatter},
    };
    use std::{
        ops::{Add, Sub},
        sync::atomic::{AtomicU64, Ordering},
    };

    struct CountFormatter;

    impl ValueFormatter for CountFormatter {
        fn scale_values(&self, _: f64, _: &mut [f64]) -> &'static str {
            "count"
        }

        fn scale_throughputs(
            &self,
            _: f64,
            throughput: &Throughput,
            values: &mut [f64],
        ) -> &'static str {
            let n = match *throughput {
                Throughput::Bits(n) => n,
                Throughput::Bytes(n) => n,
                Throughput::BytesDecimal(n) => n,
                Throughput::Elements(n) => n,
                Throughput::ElementsAndBytes { elements, .. } => elements,
            };
            let scale = (n as f64).recip();
            for v in values {
                (*v) *= scale;
            }
            "results/count"
        }

        fn scale_for_machines(&self, _: &mut [f64]) -> &'static str {
            ""
        }
    }

    pub struct CountMeasurement;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    impl CountMeasurement {
        pub fn increment() {
            COUNTER.fetch_add(1, Ordering::Release);
        }

        fn read() -> MeasurementValue {
            MeasurementValue::from(COUNTER.load(Ordering::Acquire))
        }
    }

    #[derive(Copy, Clone, Default)]
    pub struct MeasurementValue(u64, std::num::Wrapping<i32>);

    impl Add for MeasurementValue {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0, self.1 + rhs.1)
        }
    }

    impl Sub for MeasurementValue {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0, self.1 - rhs.1)
        }
    }

    impl From<u64> for MeasurementValue {
        fn from(value: u64) -> Self {
            Self(value, rand::random())
        }
    }

    impl Into<f64> for MeasurementValue {
        fn into(self) -> f64 {
            // Criterion panics if measurement variance is zero, so we introduce some synthetic
            // noise.
            self.0 as f64 + (self.1.0 as f64) / (i32::MAX as f64)
        }
    }

    impl Measurement for CountMeasurement {
        type Intermediate = MeasurementValue;
        type Value = MeasurementValue;

        fn start(&self) -> Self::Intermediate {
            Self::read()
        }

        fn end(&self, i: Self::Intermediate) -> Self::Value {
            self.start() - i
        }

        fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
            *v1 + *v2
        }

        fn zero(&self) -> Self::Value {
            Default::default()
        }

        fn to_f64(&self, &value: &Self::Value) -> f64 {
            value.into()
        }

        fn formatter(&self) -> &dyn ValueFormatter {
            &CountFormatter
        }
    }
}
