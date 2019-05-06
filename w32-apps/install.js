var WshShell = WScript.CreateObject("WScript.Shell");
strSendTo = WshShell.SpecialFolders("SendTo");
var oShellLink = WshShell.CreateShortcut(strSendTo + "\\waifu2xGUI.lnk");
oShellLink.TargetPath = WshShell.CurrentDirectory + "\\w2xcr.exe";
oShellLink.WindowStyle = 1;
oShellLink.IconLocation = WshShell.CurrentDirectory + '\\icon.ico';
oShellLink.Description = "waifu2xGUI";
oShellLink.Arguments = "--block_size 0 --interactive";
oShellLink.Save();
