package spotify_framework;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.util.Log;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.google.gson.Gson;
import com.spotify.android.appremote.api.ConnectionParams;
import com.spotify.android.appremote.api.Connector;

import com.spotify.protocol.client.Subscription;
import com.spotify.protocol.types.PlayerState;
import com.spotify.protocol.types.Track;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class PlaybackService {
    private boolean active;
    private String deviceId;
    private SharedPreferences preferences;
    private RequestQueue queue;

    public PlaybackService(Context context) {
        preferences = context.getSharedPreferences("SPOTIFY",0);
        queue = Volley.newRequestQueue(context);
    }

    public boolean getActive() {
        return active;
    }

    public String getDeviceId() {
        return deviceId;
    }

    
    public void findDevice(VolleyCallBack callBack) {
        String endpoint = "https://api.spotify.com/v1/me/player/devices";
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.GET,
                endpoint,
                null,
                new Response.Listener<JSONObject>() {
                    @Override

                    public void onResponse(JSONObject response) {
                        System.out.println("RESPONSE: " + response);
                        JSONArray devices = response.optJSONArray("devices");
                        for (int i = 0; i < devices.length(); i++) {
                            try {
                                System.out.println(devices.getString(i));
                                JSONObject device = devices.getJSONObject(i);
                                System.out.println(Build.MODEL +" :: " + device.getString("name"));
                                System.out.println(Build.MODEL.equals(device.getString("name")));
                                if (device.getString("name").equals(android.os.Build.MODEL)) {
                                    deviceId = device.getString("id");
                                }
                            } catch (JSONException e) {
                                e.printStackTrace();
                            }
                        }
                        callBack.onSuccess();
                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Log.e("Error Occured", error.toString());
            }
        }) {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<String, String>();
                String token = preferences.getString("TOKEN", "");
                String auth = "Bearer " + token;
                headers.put("Authorization", auth);
                return headers;
            }
        };
        queue.add(jsonObjectRequest);
    }

    public void addToQueue(Song song) {
        String endpoint = "https://api.spotify.com/v1/me/player/queue";
        String uri = "uri=" + song.getUri();
        String device = "device_id=" + deviceId;

        endpoint = endpoint + "?" + uri + "&" + device;

        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.POST,
                endpoint,
                null,
                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {

                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {

            }
        }) {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<String, String>();
                String token = preferences.getString("TOKEN", "");
                String auth = "Bearer " + token;
                headers.put("Authorization", auth);
                return headers;
            }
        };
           queue.add(jsonObjectRequest);
    }

    public void play() {
        String endpoint = "https://api.spotify.com/v1/me/player/play";
        String device = "device_id=" + deviceId;

        endpoint = endpoint + "?" + device;
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.PUT,
                endpoint,
                null,
                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {

                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {

            }
        })
        {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<String, String>();
                String token = preferences.getString("TOKEN", "");
                String auth = "Bearer " + token;
                headers.put("Authorization", auth);
                return headers;
            }
        };
        queue.add(jsonObjectRequest);
    }

    public void pause() {
        String endpoint = "https://api.spotify.com/v1/me/player/pause";
        String device = "device_id=" + deviceId;
        endpoint = endpoint + "?" + device;

        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.PUT,
                endpoint,
                null,
                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {

                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {

            }
        })
        {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<String, String>();
                String token = preferences.getString("TOKEN", "");
                String auth = "Bearer " + token;
                headers.put("Authorization", auth);
                headers.put("device_id", deviceId);
                return headers;
            }
        };
        queue.add(jsonObjectRequest);
    }

    public void prev() {
        String endpoint = "https://api.spotify.com/v1/me/player/previous";
        String device = "device_id=" + deviceId;
        endpoint = endpoint + "?" + device;

        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.POST,
                endpoint,
                null,
                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {

                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {

            }
        })
        {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<String, String>();
                String token = preferences.getString("TOKEN", "");
                String auth = "Bearer " + token;
                headers.put("Authorization", auth);
                headers.put("device_id", deviceId);
                return headers;
            }
        };
        queue.add(jsonObjectRequest);
    }

    public void next() {
        String endpoint = "https://api.spotify.com/v1/me/player/next";
        String device = "device_id=" + deviceId;
        endpoint = endpoint + "?" + device;

        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.POST,
                endpoint,
                null,
                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {

                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {

            }
        })
        {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<String, String>();
                String token = preferences.getString("TOKEN", "");
                String auth = "Bearer " + token;
                headers.put("Authorization", auth);
                headers.put("device_id", deviceId);
                return headers;
            }
        };
        queue.add(jsonObjectRequest);
    }
}
