var WshShell = WScript.CreateObject("WScript.Shell");
var objFSO = new ActiveXObject("Scripting.FileSystemObject");
findWaifuFolder(); //check paths

var strSendTo = WshShell.SpecialFolders("SendTo");
oneShortcut("waifu_n1",   "waifuExecuter.wsf", "--without_typing --scale_ratio 1 --noise_level 1");
oneShortcut("waifu_n2",   "waifuExecuter.wsf", "--without_typing --scale_ratio 1 --noise_level 2");
oneShortcut("waifu2x_n0", "waifuExecuter.wsf", "--without_typing --scale_ratio 2 --noise_level 0");
oneShortcut("waifu2x_n1", "waifuExecuter.wsf", "--without_typing --scale_ratio 2 --noise_level 1");
oneShortcut("waifu2x_n2", "waifuExecuter.wsf", "--without_typing --scale_ratio 2 --noise_level 2");
oneShortcut("waifuCustomizable", "waifuExecuter.wsf", "");

WScript.echo(
	"Скрипты успешно добавлены в меню \"Отправить\". Теперь Вы можете нажать правой кнопкой на любое изображение или сразу несколько изображений, выбрать пункт \"Отправить\" и кликнуть на один из вариантов:\n\nwaifu_n1 — устранить шум\nwaifu_n2 — агрессивно устранить шум\nwaifu2x_n0 — увеличить картинку в 4 раза без устранения шума\nwaifu2x_n1 — увеличить картинку в 4 раза и устранить шум\nwaifu2x_n2 — увеличить картинку в 4 раза и агрессивно устранить шум\nwaifuCustomizable — самостоятельно указать все опции" +
	"\n\n========== English: ==========\n\n" +
	"Scripts have been successfully added to \"Send To\" menu. Now you can highlight few files, click right button on them, choose \"Send To\" and click on the one of this option:\n\nwaifu_n1 — denoise\nwaifu_n2 — aggressive denoise\nwaifu2x_n0 — 4x scale\nwaifu2x_n1 — 4x scale and denoise\nwaifu2x_n2 — 4x scale and aggressive denoise\nwaifuCustomizable — specify all the options by yourself" +
	"\n\n========== 日本語 ==========\n\n" +
	"スクリプトは送るメニューを追加されました。いくつかの画像を右クリックして、「送る」アイテムを選ぶとwaifu2xで処理できます。\n\n「waifu_n1」 はノイズ除去\n「waifu_n2」 はアグレッシブノイズ除去\n「waifu2x_n0」 は2x拡大\n「waifu2x_n1」 は2x拡大とノイズ除去\n「waifu2x_n1」 は2x拡大とアグレッシブノイズ除去\n「waifuCustomizable」 はオプションを自分で指定できる\n\n追伸　私は日本語が分からなくてただの書いてみましたので下手ごめんなさい (^_^)"
);

//Create one shortcut
function oneShortcut(name, file, args) {
	var oShellLink = WshShell.CreateShortcut(strSendTo + "\\" + name + ".lnk");
	oShellLink.TargetPath = "WScript";
	oShellLink.WindowStyle = 1;
	oShellLink.Description = "waifu2x";
	oShellLink.Arguments = '"' + WshShell.CurrentDirectory + "\\" + file + '" ' + args;
	oShellLink.Save();
}

//Find waifu2x-converter folder
function findWaifuFolder() {
	var tm = WScript.ScriptFullName, x = tm.length, dir, exec;
	for (var i = 0;; i++) {
		x = tm.lastIndexOf("\\", x - 1);
		if (x === -1 || i >= 2) error("Перед установкой Вы должны поместить папку ExtendedSendTo в папку с программой waifu2x-converter.\n\nYou should put ExtendedSendTo folder to folder with waifu2x-converter program before install.\n\nインストール前にExtendedSendToフォルダーをwaifu2x-converterプログラムフォルダーに置きましてください。", 2);
		if (objFSO.FileExists(tm.substr(0, x + 1) + "waifu2x-converter_x64.exe")) return {dir: tm.substr(0, x + 1), exec: "waifu2x-converter_x64.exe"};
		if (objFSO.FileExists(tm.substr(0, x + 1) + "waifu2x-converter_x86.exe")) return {dir: tm.substr(0, x + 1), exec: "waifu2x-converter_x86.exe"};
		if (objFSO.FileExists(tm.substr(0, x + 1) + "waifu2x-converter.exe"))     return {dir: tm.substr(0, x + 1), exec: "waifu2x-converter.exe"};
	}
}

//Standart functions
function nameFyFile(s) { var x = s.lastIndexOf("\\"); if (x === -1) error("Произошла ошибка при обработке имени файла", 2); return s.substr(x + 1); }
function echo(s, opt, wait, theTitle) { return WshShell.Popup(s, wait ? wait : 0, theTitle ? theTitle : title(), opt ? opt : 0); }
function error(s, crit) { echo(s,  crit == 1 ? 48 : (crit == 2 ? 16 : 0),  0,  crit === 2 ? "Ошибка" : "");  WScript.Quit(0); }
function title() { return "Waifu2x"; }
