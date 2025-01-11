using Modding;
using UnityEngine;
using System.Diagnostics;
using System.IO.Pipes;
using System.IO;
using System.Collections;
using System;
using System.Threading;
using System.Text;

namespace HKAgentMod {
    public class HKAgentMod : Mod {
        new public string GetName() => "HKAgentMod";
        public override string GetVersion() => "v0.1.2";
        /*
         * Mod
        */
        private int width, height;
        private bool isEnabled;
        private Properties properties;
        private Process pipeProcess;
        private BackgroundTaskProcessor bgProcessor;
        /*
         * Video
        */
        private NamedPipeClientStream pipeClientStream;
        private int actualFrame = 0;
        private Texture2D texture;

        public HKAgentMod() : base() {
            isEnabled = false;
            properties = new Properties(this);
            if (!Directory.Exists(properties.getFolderPath())) {
                Log("Error: no existe la carpeta de guardado");
                Application.Quit(-1);
                throw new Exception("Error: no existe la carpeta de guardado");
            }
            height = properties.getHeightScreen();
            width = properties.getWidthScreen();
            Log(Directory.GetCurrentDirectory());
            Log(joinAbsolutePath("hollow_knight_Data/Managed/Mods/My_Agent_Hollow_Knight_Mod/utils/"));
            Log(Directory.Exists(joinAbsolutePath("hollow_knight_Data/Managed/Mods/My_Agent_Hollow_Knight_Mod/utils/")));
            pipeProcess = new Process {
                StartInfo = {
                    FileName = "python",
                    Arguments = Path.Combine(properties.getFolderPath(), "agent.py"),
                    //Arguments  = joinAbsolutePath("hollow_knight_Data/Managed/Mods/My_Agent_Hollow_Knight_Mod/utils/server.py"),
                    //Arguments = ('"'),
                    UseShellExecute = false,
                    WorkingDirectory = properties.getFolderPath()
                    //WorkingDirectory = joinAbsolutePath("hollow_knight_Data/Managed/Mods/My_Agent_Hollow_Knight_Mod/utils"),
                    //CreateNoWindow = true
                },
                EnableRaisingEvents = true
            };
            pipeProcess.Exited += new EventHandler(OnPipeExit);
            pipeProcess.Start();
            EventWaitHandle eventWaitHandle = new EventWaitHandle(false, EventResetMode.AutoReset, "MiEventoHKAgent");
            eventWaitHandle.WaitOne();
            pipeClientStream = new NamedPipeClientStream(".", "agent_processing", PipeDirection.Out, PipeOptions.Asynchronous);
            texture = new Texture2D(width, height);
            bgProcessor = new BackgroundTaskProcessor();
        }
        public override void Initialize() {
            Log("Conectando al PIPE");
            pipeClientStream.Connect();
            Log("Conectado al PIPE con exito");
            ModHooks.SavegameLoadHook += OnLoad;
            ModHooks.NewGameHook += () => OnLoad(0);
            ModHooks.HeroUpdateHook += OnHeroUpdate;
            ModHooks.ApplicationQuitHook += () => {
                if (this.isEnabled) {
                    bgProcessor.EnqueueTask(() => pipeClientStream.WriteByte(0b11000000));
                    pipeProcess.Kill();
                }
            };
            // Recording mark
            GameObject modObject = new GameObject("MarkOnRecording");
            modObject.AddComponent<MarkOnRecording>();
            UnityEngine.Object.DontDestroyOnLoad(modObject);
        }

        private void OnLoad(int _) {
            UIManager ui = UIManager.instance;
            ui.DefaultVideoMenuSettings();
            // Resolution
            Screen.SetResolution(width, height, false, 60);
            Application.targetFrameRate = 60;
            // VSYNC
            ui.vsyncOption.UpdateSetting(1);
            ui.vsyncOption.RefreshValueFromGameSettings(false);
        }
        public void OnHeroUpdate() {
            try {
                if (Input.GetKeyDown(KeyCode.O)) {
                    if (this.isEnabled)
                        this.endRecording();
                    else
                        this.beginRecording();
                }
                if (!this.isEnabled || Time.frameCount % 3 != 0)
                    return;
                if (this.actualFrame == this.properties.getBatchSize()) { // BatchSize -> Time limit
                    // reinitialize
                    actualFrame = 0;
                    bgProcessor.EnqueueTask(() => pipeClientStream.WriteByte(0b00110000));
                    generateNewFileName();

                }
                actualFrame++;
                // Video Recorder
                GameManager.instance.StartCoroutine(writeVideoFrame());
            }
            catch (Exception ex) {
                Log(ex.ToString());
                Application.Quit(-1);
            }
        }
        public IEnumerator writeVideoFrame() {
            yield return new WaitForEndOfFrame();
            texture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            texture.Apply();
            bgProcessor.EnqueueTask(() => ProcessFrame(texture));
            //Task task = Task.Run(() => ProcessFrame(texture));
        }
        private void ProcessFrame(Texture2D texTemp) {
            byte[] bytes = texTemp.EncodeToJPG();
            pipeClientStream.WriteByte(0b00001100);
            pipeClientStream.Write(bytes, 0, bytes.Length);
        }
        private void beginRecording() {
            isEnabled = true;
            actualFrame = 0;
            generateNewFileName();
        }
        private void endRecording() {
            isEnabled = false;
            bgProcessor.EnqueueTask(() => pipeClientStream.WriteByte(0b00110000));
        }
        private static string joinAbsolutePath(string path) {
            return Path.Combine(Directory.GetCurrentDirectory(), path).Replace("\\", "/");
        }
        private void generateNewFileName() {
            String fileName = this.properties.getUsername() + " " + DateTime.Now.ToString("yyyy-MM-dd HH'h'mm'm'ss's'");
            byte[] fileNameBytes = Encoding.UTF8.GetBytes(fileName);
            bgProcessor.EnqueueTask(() => {
                pipeClientStream.WriteByte(0b00000011);
                pipeClientStream.Write(fileNameBytes, 0, fileNameBytes.Length);
            });
        }
        private void OnPipeExit(object sender, EventArgs e) {
            Log(pipeProcess.ExitCode);
            Log(e.ToString());
            if (pipeProcess.ExitCode != 0) {
                Application.Quit(-1);
            }
        }
    }
}

/*
* 3 0000 0011 -> mensaje de nombre de nuevo archivo
* 12 0000 1100 -> frame
* 48 0011 0000 -> abortar video actual
* 192 1100 0000 -> cerrar el pipe
*/
