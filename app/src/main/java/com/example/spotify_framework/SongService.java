package com.example.spotify_framework;

import android.app.VoiceInteractor;
import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.google.gson.Gson;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.Thread.sleep;

public class SongService {
    private ArrayList<Song> playlist;
    private Song song;
    private SharedPreferences preferences;
    private RequestQueue queue;

    public SongService(Context context) {
        preferences = context.getSharedPreferences("SPOTIFY",0);
        queue = Volley.newRequestQueue(context);

    }

    public ArrayList<Song> getPlaylist() {
        return playlist;
    }

    public Song getSong() {return song;}

    public void getRecentlyPlayed(final VolleyCallBack callBack) {
        playlist = new ArrayList<Song>();

        String endpoint = "https://api.spotify.com/v1/me/player/recently-played";
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.GET,
                endpoint,
                null,
                response  -> {
                    System.out.println("Playlist Response: " + response.toString());
                    Gson gson = new Gson();
                    JSONArray jsonArray = response.optJSONArray("items");
                    for(int i = 0; i < jsonArray.length(); i++) {
                        try {
                            JSONObject object = jsonArray.getJSONObject(i);
                            object = object.optJSONObject("track");
                            Log.d("Song Response: ", object.toString());
                            populateSong(object.optString("id"), () -> {
                                System.out.println(getSong().getName());
                               playlist.add(getSong());
                            });
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }

                    callBack.onSuccess();
                }, error -> getRecentlyPlayed(() -> {

        })) {
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

    public void populateSong(String id, final VolleyCallBack callBack) {
        String endpoint = "https://api.spotify.com/v1/tracks/" + id;
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.GET,
                endpoint,
                null,
                response -> {
                    Song currentSong;
                    Gson gson = new Gson();
                    currentSong = gson.fromJson(response.toString(), Song.class);
                    JSONObject object = response.optJSONObject("album");
                    JSONArray images = object.optJSONArray("images");
                    for(int i = 0 ; i < images.length(); i++) {
                        try {
                            JSONObject pic = images.getJSONObject(i);
                            currentSong.setImage(pic.optString("url"));
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                    song = currentSong;
                    callBack.onSuccess();
                }, error -> populateSong(id, () -> {

        })) {
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

    public static ArrayList<Song> createSongs() {
        ArrayList<Song> songs = new ArrayList<Song>();
        for(int i = 0; i < 21; i++) {
            Song song = new Song(String.valueOf(i), "Song " + i);
            song.setArtist("Artist " + i);
            songs.add(song);
        }
        return songs;
    }
}
