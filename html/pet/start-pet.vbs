Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
Set ws = CreateObject("Wscript.Shell")
ws.CurrentDirectory = dir
ws.Run """" & dir & "\node_modules\electron\dist\electron.exe"" .", 0, False
