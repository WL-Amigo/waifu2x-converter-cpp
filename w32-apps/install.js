var WshShell = WScript.CreateObject("WScript.Shell");
strSendTo = WshShell.SpecialFolders("SendTo");
var oShellLink = WshShell.CreateShortcut(strSendTo + "\\waifu2x.lnk");
oShellLink.TargetPath = WshShell.CurrentDirectory + "\\w2xcr.exe";
oShellLink.WindowStyle = 1;
oShellLink.Description = "waifu2x";
oShellLink.Arguments = "--block_size 0";
oShellLink.Save();
