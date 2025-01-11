using System;
using System.Collections.Concurrent;
using System.Threading;

namespace HKAgentMod {
    internal class BackgroundTaskProcessor {
        private readonly BlockingCollection<Action> taskQueue = new BlockingCollection<Action>();
        private readonly Thread workerThread;

        public BackgroundTaskProcessor() {
            workerThread = new Thread(ProcessTasks) {
                IsBackground = true
            };
            workerThread.Start();
        }

        public void EnqueueTask(Action task) {
            taskQueue.Add(task);
        }

        private void ProcessTasks() {
            foreach (var task in taskQueue.GetConsumingEnumerable()) {
                try {
                    task();
                }
                catch (Exception ex) {
                    Console.WriteLine($"Error executing task: {ex.Message}");
                }
            }
        }

        public void StopProcessing() {
            taskQueue.CompleteAdding();
            workerThread.Join();
        }
    }
}
