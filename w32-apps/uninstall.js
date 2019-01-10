var WshShell = WScript.CreateObject("WScript.Shell");
var FSO = WScript.CreateObject("Scripting.FileSystemObject");
strSendTo = WshShell.SpecialFolders("SendTo");
FSO.DeleteFile(strSendTo + "\\waifu2xGUI.lnk");
