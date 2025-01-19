using System.Collections.Generic;
using System;
using System.IO;
using System.Diagnostics;

namespace HKAgentMod {
    internal class Properties {
        private Dictionary<string, string> properties;
        public Properties(HKAgentMod mod) {
            properties= new Dictionary<string, string>();
            try {
                foreach (string line in File.ReadAllLines("hollow_knight_Data/Managed/Mods/My_Agent_Hollow_Knight_Mod/config.ini")) {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#") || !line.Contains("="))
                        continue;
                    mod.Log(line);
                    string[] parts = line.Split(new char[] { '=' }, 2);
                    if (parts.Length == 2) {
                        string key = parts[0].Trim();
                        string value = parts[1].Trim();
                        mod.Log(key + " - " + value);
                        properties[key] = value;
                    }
                }
            }
            catch (Exception ex) {
                mod.Log(ex.Message);
                Debug.Assert(false);
                throw ex;
            }
        }

        public override String ToString() {
            String ret = "";
            foreach (string key in properties.Keys) {
                ret += key + "=" + properties[key] + "\n";
            }
            return ret;
        }
        public String getFolderPath() {
            return this.properties["FolderPath"].Replace("\\", "/");            
        }
        public int getWidthScreen() {
            return Int32.Parse(this.properties["ScreenWidth"]);
        }
        public int getHeightScreen() {
            return Int32.Parse(this.properties["ScreenHeight"]);
        }
    }
}