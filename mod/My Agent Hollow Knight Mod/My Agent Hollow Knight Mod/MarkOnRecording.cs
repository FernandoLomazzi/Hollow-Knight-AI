using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace HKAgentMod {
    internal class MarkOnRecording : MonoBehaviour{
        private bool showSquare = false;
        private Texture2D squareTexture = new Texture2D(size, size, TextureFormat.RGBA32, false);
        private const int size = 10;
        private const int padding = 0;

        public void Start() {
            for (int y = 0; y < squareTexture.height; y++) {
                for (int x = 0; x < squareTexture.width; x++) {
                    squareTexture.SetPixel(x, y, Color.red);
                }
            }
            squareTexture.Apply();
        }
        public void Update() {
            if (Input.GetKeyDown(KeyCode.O)) {
                showSquare = !showSquare;
            }
        }

        public void OnGUI() {
            if (showSquare) {
                GUI.DrawTexture(new Rect(Screen.width - padding - size, padding, size, size), squareTexture);
            }
        }
    }
}
