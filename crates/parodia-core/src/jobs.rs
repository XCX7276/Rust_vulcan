//! Пул рабочих потоков с очередью по приоритету.
//!
//! Зачем: генерация и построение моделей чанков занимают миллисекунды на чанк. Если делать это
//! в главном потоке, игра дёргается. Поэтому тяжёлую работу выполняют несколько постоянных
//! рабочих потоков, а главный поток только раздаёт задания и забирает готовые результаты.
//!
//! Как устроено:
//! - потоки создаются один раз при старте (не на каждый чанк);
//! - задания лежат в общей очереди, первым берётся задание с наименьшим `priority`
//!   (например, ближайший к камере чанк); при равном приоритете — то, что пришло раньше;
//! - у каждого потока своё рабочее «состояние» `S` (например, свой `Mesher` с буферами) —
//!   буферы переиспользуются, а потоки не мешают друг другу;
//! - результаты складываются в канал, главный поток забирает их без ожидания (`try_recv`);
//! - если задание упало с паникой, поток не умирает: ошибка пишется в журнал.
//!
//! Очередь не ограничивается здесь — ограничение (сколько заданий «в полёте») держит тот,
//! кто отправляет задания (стример), чтобы при быстром полёте не копились устаревшие задачи.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Job<S, R> = Box<dyn FnOnce(&mut S) -> R + Send>;

struct Entry<S, R> {
    priority: f32,
    sequence: u64,
    job: Job<S, R>,
}

impl<S, R> PartialEq for Entry<S, R> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<S, R> Eq for Entry<S, R> {}
impl<S, R> PartialOrd for Entry<S, R> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<S, R> Ord for Entry<S, R> {
    // BinaryHeap отдаёт наибольший элемент — поэтому «меньший приоритет» считаем «большим».
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .total_cmp(&self.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

struct Queue<S, R> {
    heap: BinaryHeap<Entry<S, R>>,
    next_sequence: u64,
    shutdown: bool,
}

struct Shared<S, R> {
    queue: Mutex<Queue<S, R>>,
    wake: Condvar,
}

pub struct WorkerPool<S, R> {
    shared: Arc<Shared<S, R>>,
    results: Receiver<R>,
    threads: Vec<JoinHandle<()>>,
}

impl<S: 'static, R: Send + 'static> WorkerPool<S, R> {
    /// Запускает `threads` потоков. `make_state` вызывается в каждом потоке один раз.
    pub fn new(
        threads: usize,
        name: &str,
        make_state: impl Fn() -> S + Send + Sync + 'static,
    ) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                heap: BinaryHeap::new(),
                next_sequence: 0,
                shutdown: false,
            }),
            wake: Condvar::new(),
        });
        let (sender, results) = channel();
        let make_state = Arc::new(make_state);
        let mut handles = Vec::with_capacity(threads);
        for index in 0..threads.max(1) {
            let shared = Arc::clone(&shared);
            let sender: Sender<R> = sender.clone();
            let make_state = Arc::clone(&make_state);
            let handle = std::thread::Builder::new()
                .name(format!("{name}-{index}"))
                .spawn(move || worker_loop(&shared, &sender, make_state()))?;
            handles.push(handle);
        }
        Ok(Self {
            shared,
            results,
            threads: handles,
        })
    }

    /// Ставит задание в очередь. Меньший `priority` выполняется раньше.
    pub fn submit(&self, priority: f32, job: impl FnOnce(&mut S) -> R + Send + 'static) {
        if let Ok(mut queue) = self.shared.queue.lock() {
            let sequence = queue.next_sequence;
            queue.next_sequence += 1;
            queue.heap.push(Entry {
                priority,
                sequence,
                job: Box::new(job),
            });
            self.shared.wake.notify_one();
        }
    }

    /// Готовый результат, если есть (не ждёт).
    pub fn try_recv(&self) -> Option<R> {
        self.results.try_recv().ok()
    }

    /// Сколько заданий ждут в очереди (ещё не взяты потоками).
    pub fn queued(&self) -> usize {
        self.shared.queue.lock().map_or(0, |q| q.heap.len())
    }

    pub fn threads(&self) -> usize {
        self.threads.len()
    }
}

fn worker_loop<S, R>(shared: &Shared<S, R>, results: &Sender<R>, mut state: S) {
    loop {
        let job = {
            let Ok(mut queue) = shared.queue.lock() else {
                return;
            };
            loop {
                if queue.shutdown {
                    return;
                }
                if let Some(entry) = queue.heap.pop() {
                    break entry.job;
                }
                queue = match shared.wake.wait(queue) {
                    Ok(queue) => queue,
                    Err(_) => return,
                };
            }
        };
        match catch_unwind(AssertUnwindSafe(|| job(&mut state))) {
            Ok(result) => {
                if results.send(result).is_err() {
                    return; // пул уничтожен
                }
            }
            Err(_) => log::error!("worker job panicked; the job's result is lost"),
        }
    }
}

impl<S, R> Drop for WorkerPool<S, R> {
    fn drop(&mut self) {
        if let Ok(mut queue) = self.shared.queue.lock() {
            queue.shutdown = true;
            queue.heap.clear();
        }
        self.shared.wake.notify_all();
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Сколько рабочих потоков взять по умолчанию: логических ядер минус 2 (главный поток и система),
/// не меньше 1 и не больше 6.
pub fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .saturating_sub(2)
        .clamp(1, 6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn collect<S: 'static, R: Send + 'static>(pool: &WorkerPool<S, R>, count: usize) -> Vec<R> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut out = Vec::new();
        while out.len() < count && Instant::now() < deadline {
            match pool.try_recv() {
                Some(r) => out.push(r),
                None => std::thread::sleep(Duration::from_millis(1)),
            }
        }
        out
    }

    #[test]
    fn runs_all_jobs_with_worker_state() {
        let pool = WorkerPool::new(3, "test", || 10u64).unwrap();
        for i in 0..100u64 {
            pool.submit(i as f32, move |state| {
                *state += 1;
                i * 2
            });
        }
        let mut results = collect(&pool, 100);
        results.sort_unstable();
        assert_eq!(results, (0..100).map(|i| i * 2).collect::<Vec<_>>());
    }

    #[test]
    fn lower_priority_value_runs_first() {
        // Один поток, первый job держит поток, пока в очередь кладутся остальные.
        let pool = WorkerPool::new(1, "test", || ()).unwrap();
        let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
        pool.submit(0.0, move |_| {
            let _ = gate_rx.recv_timeout(Duration::from_secs(5));
            -1
        });
        std::thread::sleep(Duration::from_millis(20));
        for p in [5, 1, 4, 2, 3] {
            pool.submit(p as f32, move |_| p);
        }
        gate_tx.send(()).unwrap();
        let order = collect(&pool, 6);
        assert_eq!(order, vec![-1, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn panicking_job_does_not_kill_worker() {
        let pool = WorkerPool::new(1, "test", || ()).unwrap();
        pool.submit(0.0, |_| -> i32 { panic!("boom") });
        pool.submit(1.0, |_| 7);
        assert_eq!(collect(&pool, 1), vec![7]);
    }
}
