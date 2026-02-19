package com.example.capstone403

import android.Manifest
import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import java.net.HttpURLConnection
import java.net.URL
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader




class MainActivity : AppCompatActivity() {

    // Robot network info
    private val ROBOT_IP = "10.244.59.219"   //"192.168.0.56" for PI
    private val ROBOT_PORT = 5000

    //private fun robotStatusUrl(): String = "http://$ROBOT_IP:$ROBOT_PORT/status"

    //private fun robotUrl(): String = "http://$ROBOT_IP:$ROBOT_PORT/"

    private fun robotStatusUrl(): String = "https://megacephalic-tarsha-dextrorotatory.ngrok-free.dev/status"

    private fun robotUrl(): String = "https://megacephalic-tarsha-dextrorotatory.ngrok-free.dev"

    // Polling / notification state
    private var pollingEnabled = true
    private var pollingThread: Thread? = null
    @Volatile private var notificationSent = false

    companion object {
        private const val CHANNEL_ID = "robot_stream_channel"
        private const val NOTIFICATION_ID = 2001
        private const val REQ_POST_NOTIFICATIONS = 1001
    }

    private fun ensureNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val granted = ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED

            if (!granted) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    REQ_POST_NOTIFICATIONS
                )
            }
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channelName = "Robot Stream Alerts"
            val channel = NotificationChannel(
                CHANNEL_ID,
                channelName,
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "Alerts when the robot stream is available"
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    private fun sendRobotStreamNotification(streamUrl: String) {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("Robot camera online")
            .setContentText("Tap to view the stream at $streamUrl")
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)

        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, builder.build())
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Permissions + notification channel
        ensureNotificationPermission()
        createNotificationChannel()

        val btnConnect = findViewById<Button>(R.id.btnConnect)
        val webView = findViewById<WebView>(R.id.webView)
        val switchPolling = findViewById<Switch>(R.id.switchPolling)
        val ipInput = findViewById<EditText>(R.id.ipInput)

        // Show the full URL (or just IP if you prefer)
        ipInput.setText(robotUrl())

        webView.settings.javaScriptEnabled = true
        webView.webViewClient = WebViewClient()

        switchPolling.setOnCheckedChangeListener { _, isChecked ->
            pollingEnabled = isChecked
            notificationSent = false
            if (isChecked) {
                startPolling(robotStatusUrl())
            } else {
                stopPolling()
            }
        }

        btnConnect.setOnClickListener {
            val url = robotUrl()
            webView.loadUrl(url)
            if (pollingEnabled) {
                startPolling(robotStatusUrl())
            }
        }

    }

    private fun startPolling(statusUrl: String) {
        stopPolling() // ensure only one thread
        pollingThread = Thread {
            println("DEBUG: polling thread started, url=$statusUrl")
            while (!Thread.currentThread().isInterrupted && pollingEnabled) {
                try {
                    println("DEBUG: opening connection...")
                    val url = URL(statusUrl)
                    val conn = url.openConnection() as HttpURLConnection
                    conn.connectTimeout = 2000
                    conn.readTimeout = 2000
                    conn.requestMethod = "GET"

                    val code = conn.responseCode
                    println("DEBUG: responseCode=$code")

                    if (code in 200..399) {
                        val reader = BufferedReader(InputStreamReader(conn.inputStream))
                        val response = StringBuilder()
                        var line: String? = reader.readLine()
                        while (line != null) {
                            response.append(line)
                            line = reader.readLine()
                        }
                        reader.close()
                        println("DEBUG: rawResponse=${response.toString()}")

                        val json = JSONObject(response.toString())
                        val shouldNotify = json.optBoolean("should_notify", false)
                        val message = json.optString("message", "BABY MONITOR ALERT")
                        val confidence = json.optDouble("confidence_score", 0.0)

                        println("DEBUG: parsed shouldNotify=$shouldNotify, confidence=$confidence")

                        if (shouldNotify && !notificationSent) {
                            println("DEBUG: triggering notification")
                            notificationSent = true
                            runOnUiThread {
                                sendRobotStreamNotification(robotUrl(), message, confidence)
                            }
                        }
                    } else {
                        println("DEBUG: non-2xx code, skipping body")
                    }
                    conn.disconnect()
                } catch (e: Exception) {
                    println("DEBUG: exception in polling: ${e.message}")
                }

                try {
                    Thread.sleep(5000) // 5s interval
                } catch (_: InterruptedException) {
                    Thread.currentThread().interrupt()
                }
            }
            println("DEBUG: polling thread exiting")
        }.apply { start() }
    }





    private fun sendRobotStreamNotification(streamUrl: String, message: String, confidence: Double) {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val contentText = "$message (confidence: ${(confidence * 100).toInt()}%)"

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("Baby monitor alert")
            .setContentText(contentText)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)

        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, builder.build())
    }



    private fun stopPolling() {
        pollingThread?.interrupt()
        pollingThread = null
    }

}
