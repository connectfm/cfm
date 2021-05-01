package spotify_framework;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.util.Log;
import android.widget.ImageView;
import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.google.gson.Gson;
import java.io.InputStream;
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

	public static ArrayList<Song> createSongs() {
		ArrayList<Song> songs = new ArrayList<Song>();
		for (int i = 0; i < 21; i++) {
			Song song = new Song(String.valueOf(i), "Song " + i);
			song.setArtist("Artist " + i);
			songs.add(song);
		}
		return songs;
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
								System.out.println(artists.length());
								for (int j = 0; j < artists.length(); j++) {
									JSONObject artist = artists.getJSONObject(j);
									s.setArtist(artist.getString("name"));
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
								JSONObject object = trackInfo.optJSONObject("album");
								s.setAlbumName(object.optString("name"));
								JSONArray images = object.optJSONArray("images");
								for (int j = 0; j < images.length(); j++) {
									JSONObject pic = images.getJSONObject(j);
									s.setImage(pic.optString("url"));
								}

								JSONArray artists = object.getJSONArray("artists");
								System.out.println(artists.length());
								for (int j = 0; j < artists.length(); j++) {
									JSONObject artist = artists.getJSONObject(j);
									s.setArtist(artist.getString("name"));
								}
								playlist.add(s);
							}
						} catch (JSONException e) {
							e.printStackTrace();
						} finally {
							removeDupes(playlist);
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

	public void removeDupes(ArrayList<Song> playlist) {
		ArrayList<String> idList = new ArrayList<>();

		for (int i = 0; i < playlist.size(); i++) {
			if (idList.contains(playlist.get(i).getId())) {
				playlist.remove(playlist.get(i));
			} else {
				idList.add(playlist.get(i).getId());
			}
		}
	}

	private class DownloadImageTask extends AsyncTask<String, Void, Bitmap> {

		ImageView bmImage;

		public DownloadImageTask(ImageView bmImage) {
			this.bmImage = bmImage;
		}

		protected Bitmap doInBackground(String... urls) {
			String urldisplay = urls[0];
			Bitmap mIcon11 = null;
			try {
				InputStream in = new java.net.URL(urldisplay).openStream();
				mIcon11 = BitmapFactory.decodeStream(in);
			} catch (Exception e) {
				Log.e("Error", e.getMessage());
				e.printStackTrace();
			}
			return mIcon11;
		}

		protected void onPostExecute(Bitmap result) {
			bmImage.setImageBitmap(result);
		}
	}
}