var WshShell = WScript.CreateObject("WScript.Shell");
strDesktop = WshShell.SpecialFolders("SendTo");
var oShellLink = WshShell.CreateShortcut(strDesktop + "\\waifu2x.lnk");
oShellLink.TargetPath = WshShell.CurrentDirectory + "\\w2xcr.exe";
oShellLink.WindowStyle = 1;
oShellLink.Description = "waifu2x";
oShellLink.Save();
