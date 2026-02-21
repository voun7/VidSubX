import logging
import time
from threading import Thread

from jeepney import DBusAddress, new_method_call, low_level
from jeepney.io.blocking import open_dbus_connection

logger = logging.getLogger(__name__)


def connection_worker(message: low_level.Message, timeout: int) -> None:
    connection = open_dbus_connection(bus='SESSION')
    reply = connection.send_and_get_reply(message)
    logger.debug(f"Notification ID: {reply.body[0]}")
    time.sleep(timeout)  # Needed to stop the notification from closing immediately when called from the gui
    connection.close()
    logger.debug(f"Notification connection closed")


def send_linux_notification(app_id: str, title: str, msg: str = "", icon: str = "", timeout: int = 8) -> None:
    notifications = DBusAddress('/org/freedesktop/Notifications',
                                bus_name='org.freedesktop.Notifications',
                                interface='org.freedesktop.Notifications')
    hints = {"sound-name": ("s", "message-new-instant")}
    message = new_method_call(notifications, 'Notify', 'susssasa{sv}i',
                              (app_id,  # App name
                               0,  # Not replacing any previous notification
                               icon,  # Icon
                               title,  # Summary
                               msg,
                               [], hints,  # Actions, hints
                               -1,  # expire_timeout (-1 = default)
                               ))
    try:
        Thread(target=connection_worker, args=(message, timeout), daemon=True).start()
    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")


if __name__ == '__main__':
    send_linux_notification(app_id='Test App', title='Test title', msg='Test message')
