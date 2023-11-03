# Source: https://pypi.org/project/winotify/
import subprocess


class Sound:
    Default = "ms-winsoundevent:Notification.Default"
    IM = "ms-winsoundevent:Notification.IM"
    Mail = "ms-winsoundevent:Notification.Mail"
    Reminder = "ms-winsoundevent:Notification.Reminder"
    SMS = "ms-winsoundevent:Notification.SMS"
    LoopingAlarm = "ms-winsoundevent:Notification.Looping.Alarm"
    LoopingAlarm2 = "ms-winsoundevent:Notification.Looping.Alarm2"
    LoopingAlarm3 = "ms-winsoundevent:Notification.Looping.Alarm3"
    LoopingAlarm4 = "ms-winsoundevent:Notification.Looping.Alarm4"
    LoopingAlarm6 = "ms-winsoundevent:Notification.Looping.Alarm6"
    LoopingAlarm8 = "ms-winsoundevent:Notification.Looping.Alarm8"
    LoopingAlarm9 = "ms-winsoundevent:Notification.Looping.Alarm9"
    LoopingAlarm10 = "ms-winsoundevent:Notification.Looping.Alarm10"
    LoopingCall = "ms-winsoundevent:Notification.Looping.Call"
    LoopingCall2 = "ms-winsoundevent:Notification.Looping.Call2"
    LoopingCall3 = "ms-winsoundevent:Notification.Looping.Call3"
    LoopingCall4 = "ms-winsoundevent:Notification.Looping.Call4"
    LoopingCall5 = "ms-winsoundevent:Notification.Looping.Call5"
    LoopingCall6 = "ms-winsoundevent:Notification.Looping.Call6"
    LoopingCall7 = "ms-winsoundevent:Notification.Looping.Call7"
    LoopingCall8 = "ms-winsoundevent:Notification.Looping.Call8"
    LoopingCall9 = "ms-winsoundevent:Notification.Looping.Call9"
    LoopingCall10 = "ms-winsoundevent:Notification.Looping.Call10"
    Silent = "silent"

    @staticmethod
    def all_sounds() -> list:
        """
        :return: All the sound variables in the class.
        """
        return [attr for attr in dir(Sound) if not callable(getattr(Sound, attr)) and not attr.startswith("__")]

    @staticmethod
    def get_sound_value(sound_name: str) -> str:
        """
        :return: Value of sound variable.
        """
        for attr in dir(Sound):
            if sound_name == attr:
                return getattr(Sound, attr)


WIN_TOAST_TEMPLATE = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$Template = @"
<toast duration="{duration}">
    <visual>
        <binding template="ToastImageAndText02">
            <image id="1" src="{icon}" />
            <text id="1"><![CDATA[{title}]]></text>
            <text id="2"><![CDATA[{msg}]]></text>
        </binding>
    </visual>
    {audio}
</toast>
"@

$SerializedXml = New-Object Windows.Data.Xml.Dom.XmlDocument
$SerializedXml.LoadXml($Template)

$Toast = [Windows.UI.Notifications.ToastNotification]::new($SerializedXml)
$Toast.Tag = "{tag}"
$Toast.Group = "{group}"

$Notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("{app_id}")
$Notifier.Show($Toast);
"""


class Notification:
    def __init__(self, app_id: str, title: str, msg: str = "", icon: str = "", duration: str = 'short') -> None:
        """
        Construct a new notification
        Args:
            app_id: your app name, make it readable to your user. It can contain spaces, however special characters
                    (e.g. é) are not supported.
            title: The heading of the toast.
            msg: The content/message of the toast.
            icon: An optional path to an image to display on the left of the title & message.
                  Make sure the path is absolute.
            duration: How long the toast should show up for (short/long), default is short.
        Raises:
            ValueError: If the duration specified is not short or long
        """
        self.app_id = app_id
        self.title = title
        self.msg = msg
        self.icon = icon
        self.duration = duration
        self.audio = Sound.Silent
        self.tag = self.title
        self.group = self.app_id
        self.script = ""
        if duration not in ("short", "long"):
            raise ValueError("Duration is not 'short' or 'long'")

    def set_audio(self, sound: str, loop: bool) -> None:
        """
        Set the audio for the notification.
        Args:
            sound: The audio to play when the notification is showing. Choose one from Sound class,
                   (eg. Sound.Default). The default for all notification is silent.
            loop: If True, the audio will play indefinitely until user click or dismiss the notification.
        """
        self.audio = f'<audio src="{sound}" loop="{str(loop).lower()}" />'

    def show(self) -> None:
        """
        Show the toast
        """
        if Sound.Silent in self.audio:
            self.audio = '<audio silent="true" />'

        self.script = WIN_TOAST_TEMPLATE.format(**self.__dict__)
        self.run_no_console_command(self.script)

    @staticmethod
    def run_no_console_command(script: str) -> None:
        """
        Run subprocess commands without relying on a console being available
        """
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        command = ["powershell.exe", "-ExecutionPolicy", "Bypass", '-Command', script]
        sub_arg = subprocess.PIPE
        subprocess.Popen(command, startupinfo=startupinfo, stdout=sub_arg, stderr=sub_arg, stdin=sub_arg)

    def clear(self) -> None:
        """
        Clear all notification created by this module from action center.
        """
        script = f"""\
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
        [Windows.UI.Notifications.ToastNotificationManager]::History.Clear('{self.app_id}')
        """
        self.run_no_console_command(script)


if __name__ == '__main__':
    test_toast = Notification(
        app_id="Test id",
        title="test title",
        msg="hey toast"
    )
    test_toast.clear()
    test_toast.set_audio(Sound.Default, loop=False)
    test_toast.show()
