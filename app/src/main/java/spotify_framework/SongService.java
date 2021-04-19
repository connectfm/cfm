package spotify_framework;

import android.app.VoiceInteractor;
import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
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
                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {
                        try {
                            JSONArray jsonArray = response.optJSONArray("items");
                            for (int i = 0; i < jsonArray.length(); i++) {
                                JSONObject song = jsonArray.getJSONObject(i);
                                Log.d("Song " + i + " JSON Script", song.toString());
                                populateSong(song.optString("id"), new VolleyCallBack() {
                                    @Override
                                    public void onSuccess() {
                                        playlist.add(getSong());
                                    }
                                });
                            }
                            callBack.onSuccess();
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                 Log.e("Mistakes were made", error.getMessage());
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

    public void populateSong(String id, final VolleyCallBack callBack) {
        String endpoint = "https://api.spotify.com/v1/tracks/" + id;
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.GET,
                endpoint,
                null,
                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {
                        try {
                            Song currentSong;
                            Gson gson = new Gson();
                            currentSong = gson.fromJson(response.toString(), Song.class);
                            JSONObject object = response.optJSONObject("album");
                            JSONArray images = object.optJSONArray("images");
                            for (int i = 0; i < images.length(); i++) {
                                JSONObject pic = images.getJSONObject(i);
                                currentSong.setImage(pic.optString("url"));
                            }
                            callBack.onSuccess();
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Log.e("Mistakes were made", error.getMessage());
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
