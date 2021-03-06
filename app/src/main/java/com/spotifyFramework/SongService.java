package com.spotifyFramework;

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
import com.google.gson.JsonArray;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class SongService {

	private final SharedPreferences preferences;
	private final RequestQueue queue;
	private ArrayList<Song> playlist;
	private Song song;

	public SongService(Context context) {
		preferences = context.getSharedPreferences("SPOTIFY", 0);
		queue = Volley.newRequestQueue(context);

	}

	public ArrayList<Song> getPlaylist() {
		return playlist;
	}

	public Song getSong() {
		return song;
	}

	public void getNewReleases(final VolleyCallBack callBack) {
		String endpoint = "https://api.spotify.com/v1/browse/new-releases";
		JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
				Request.Method.GET,
				endpoint,
				null,
				new Response.Listener<JSONObject>() {
					@Override
					public void onResponse(JSONObject response) {
						try {
							JSONObject songs = response.optJSONObject("albums");
							JSONArray jsonArray = songs.optJSONArray("items");
							Gson gson = new Gson();
							for (int i = 0; i < jsonArray.length(); i++) {
								JSONObject song = jsonArray.getJSONObject(i);
								JSONObject trackInfo = song.getJSONObject("track");
								Song s = gson.fromJson(trackInfo.toString(), Song.class);
								JSONObject object = trackInfo.optJSONObject("album");
								s.setAlbumName(object.optString("name"));
								JSONArray images = object.optJSONArray("images");
								for (int j = 0; j < images.length(); j++) {
									JSONObject pic = images.getJSONObject(j);
									s.setImage(pic.optString("url"));
								}

								JSONArray artists = object.getJSONArray("artists");
								for (int j = 0; j < artists.length(); j++) {
									JSONObject artist = artists.getJSONObject(j);
									s.setArtist(artist.getString("id"),artist.getString("name"));
								}
								playlist.add(s);
							}
						} catch (JSONException e) {
							e.printStackTrace();
						} finally {
							callBack.onSuccess();
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
							Gson gson = new Gson();
							for (int i = 0; i < jsonArray.length(); i++) {
								JSONObject song = jsonArray.getJSONObject(i);
								JSONObject trackInfo = song.getJSONObject("track");
								Song s = gson.fromJson(trackInfo.toString(), Song.class);
								getFeatures(s);
								JSONObject object = trackInfo.optJSONObject("album");
								s.setAlbumName(object.optString("name"));
								JSONArray images = object.optJSONArray("images");
								for (int j = 0; j < images.length(); j++) {
									JSONObject pic = images.getJSONObject(j);
									s.setImage(pic.optString("url"));
								}

								JSONArray artists = object.getJSONArray("artists");
								for (int j = 0; j < artists.length(); j++) {
									JSONObject artist = artists.getJSONObject(j);
									s.setArtist(artist.getString("id"),artist.getString("name"));

								}
								playlist.add(s);
							}
						} catch (JSONException e) {
							e.printStackTrace();
						} finally {

							callBack.onSuccess();
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

	public void populateMultipleSongs(ArrayList<String> ids, VolleyCallBack callBack) {
		String endpoint = "https://api.spotify.com/v1/tracks?ids=";
		playlist = new ArrayList<Song>();

		for (int i = 0; i < ids.size(); i++) {
			if (i == ids.size() - 1) {
				endpoint = endpoint + ids.get(i) + "&market=ES";
			} else {
				endpoint = endpoint + ids.get(i) + "%2C";
			}
		}

		JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
				Request.Method.GET,
				endpoint,
				null,
				new Response.Listener<JSONObject>() {
					@Override
					public void onResponse(JSONObject response) {
						System.out.println("SYSTEM RESPONSE:: " + response.toString());
						try {
							Gson gson = new Gson();
							JSONArray songs = response.optJSONArray("tracks");
							for (int i = 0; i < songs.length(); i++) {
								JSONObject looking = songs.getJSONObject(i);
								Song current = gson.fromJson(looking.toString(), Song.class);
								JSONObject object = looking.optJSONObject("album");
								JSONArray images = object.optJSONArray("images");
								JSONArray artists = object.optJSONArray("artists");
								for (int j = 0; j < images.length(); j++) {
									JSONObject pic = images.getJSONObject(j);
									current.setImage(pic.optString("url"));
								}

								for(int j = 0; j < artists.length(); j++) {
									JSONObject artist = artists.getJSONObject(j);
									current.setArtist(artist.getString("id"),artist.getString("name"));
								}
								playlist.add(current);
							}
						} catch (JSONException e) {
							e.printStackTrace();
						} finally {
							callBack.onSuccess();
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

							Gson gson = new Gson();
							song = gson.fromJson(response.toString(), Song.class);
							JSONObject object = response.optJSONObject("album");
							JSONArray images = object.optJSONArray("images");
							for (int i = 0; i < images.length(); i++) {
								JSONObject pic = images.getJSONObject(i);
								song.setImage(pic.optString("url"));
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
	
	public void getFeatures(Song current) {
		String endpoint = "https://api.spotify.com/v1/audio-features/" + current.getId();
		JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
				Request.Method.GET,
				endpoint,
				null,
				new Response.Listener<JSONObject>() {
					@Override
					public void onResponse(JSONObject response) {
						try {
							current.setFeatures(response);
						} catch (JSONException e) {
							e.printStackTrace();
						}
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

}

