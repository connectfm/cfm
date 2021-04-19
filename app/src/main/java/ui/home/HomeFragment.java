package ui.home;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.example.cfm.R;
import ui.LoginActivity;
import com.example.spotify_framework.Playlist;
import com.example.spotify_framework.Song;
import com.example.spotify_framework.SongService;
import com.example.spotify_framework.VolleyCallBack;
import com.google.gson.Gson;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static java.lang.Thread.sleep;

public class HomeFragment extends Fragment {
    public SongService songService;
    private RecyclerView recyclerView;
    private SongsAdapter adapter;
    private ArrayList songs;

    private HomeViewModel homeViewModel;

    public void onCreate() {
    }

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        View root = inflater.inflate(R.layout.fragment_home, container, false);
        return root;
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        SharedPreferences preferences = getActivity().getSharedPreferences("SPOTIFY", 0);
        RequestQueue queue = Volley.newRequestQueue(getActivity());
            songs = new ArrayList<Song>();

            String endpoint = "https://api.spotify.com/v1/me/player/recently-played";
            JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                    Request.Method.GET,
                    endpoint,
                    null,
                    new Response.Listener<JSONObject>() {
                        @Override
                        public void onResponse(JSONObject response) {
                            try {
                                Gson gson = new Gson();
                                JSONArray jsonArray = response.optJSONArray("items");
                                for (int i = 0; i < jsonArray.length(); i++) {
                                    JSONObject song = jsonArray.getJSONObject(i);
                                    Log.d("Song " + i + " JSON Script", song.toString());
                                    Song s = gson.fromJson(song.toString(), Song.class);
                                    JSONObject trackInfo = song.getJSONObject("track");
                                    JSONObject object = trackInfo.optJSONObject("album");
                                    JSONArray images = object.optJSONArray("images");
                                    for (int j = 0; j < images.length(); j++) {
                                        JSONObject pic = images.getJSONObject(j);
                                        s.setImage(pic.optString("url"));
                                    }

                                    JSONArray artists = object.getJSONArray("artists");
                                    System.out.println(artists.length());
                                    for(int j = 0; j < artists.length(); j++) {
                                        JSONObject artist = artists.getJSONObject(j);
                                        s.setArtist(artist.getString("name"));
                                    }
                                    songs.add(s);
                                }

                            } catch (JSONException e) {
                                e.printStackTrace();
                            } finally {
                                adapter = new SongsAdapter(getActivity(), songs);
                                adapter.notifyDataSetChanged();
                                RecyclerView recyclerView = (RecyclerView) getActivity().findViewById(R.id.listening_history);
                                recyclerView.setAdapter(adapter);
                                recyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
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
    private View waitForSongInfo(final ArrayList<Song> songs, View view, RecyclerView recyclerView,Context context) {
        songService = new SongService(context);
        songService.getRecentlyPlayed(() -> {
            for(Song s: songService.getPlaylist()) {
                songs.add(s);
            }
            adapter = new SongsAdapter(getActivity(), songs);
        });
        return view;
    }

}