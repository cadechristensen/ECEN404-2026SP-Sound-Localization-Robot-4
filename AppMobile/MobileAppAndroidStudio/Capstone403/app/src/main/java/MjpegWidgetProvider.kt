package com.example.capstone403

import com.example.capstone403.R
import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.StrictMode
import android.widget.RemoteViews
import java.net.HttpURLConnection
import java.net.URL

class MjpegWidgetProvider : AppWidgetProvider() {
    override fun onUpdate(context: Context, appWidgetManager: AppWidgetManager, appWidgetIds: IntArray) {
        for (appWidgetId in appWidgetIds) {
            val views = RemoteViews(context.packageName, R.layout.widget_mjpeg)

            // Workaround to allow network in main thread for demo.
            StrictMode.setThreadPolicy(StrictMode.ThreadPolicy.Builder().permitNetwork().build())
            try {
                // Fetch a snapshot from MJPEG stream
                // Replace with your server IP
                val url = URL("http://192.168.1.100:5000/video/snapshot.jpg")
                val conn = url.openConnection() as HttpURLConnection
                val bmp = BitmapFactory.decodeStream(conn.inputStream)
                views.setImageViewBitmap(R.id.widgetImage, bmp)
            } catch (e: Exception) {
                // On error, use a placeholder
                views.setImageViewResource(R.id.widgetImage, android.R.drawable.ic_dialog_alert)
            }

            // Optional: Make tap open MainActivity
            val intent = Intent(context, MainActivity::class.java)
            val pendingIntent = PendingIntent.getActivity(context, 0, intent, PendingIntent.FLAG_IMMUTABLE)
            views.setOnClickPendingIntent(R.id.widgetImage, pendingIntent)

            appWidgetManager.updateAppWidget(appWidgetId, views)
        }
    }
}
