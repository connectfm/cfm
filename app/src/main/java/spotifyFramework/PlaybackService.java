package spotifyFramework;

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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class PlaybackService {

	private final SharedPreferences preferences;
	private final RequestQueue queue;
	private Song current;
	private boolean active;
	private int progress;
	private String deviceId;

	public PlaybackService(Context context) {
		preferences = context.getSharedPreferences("SPOTIFY", 0);
		queue = Volley.newRequestQueue(context);
	}

	public boolean getActive() {
		return active;
	}

	public String getDeviceId() {
		return deviceId;
	}

	public Song getCurrentlyPlaying() {return current;}

	public int getProgress() { return progress; }

	public void currentlyPlaying(VolleyCallBack callBack) {
		String endpoint = "https://api.spotify.com/v1/me/player/currently-playing";

		JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
				Request.Method.GET,
				endpoint,
				null,
				new Response.Listener<JSONObject>() {
					@Override
					public void onResponse(JSONObject response) {
						try {

							Gson gson = new Gson();
							current = gson.fromJson(response.toString(), Song.class);
							progress = response.getInt("progress_ms");
						} catch (JSONException e) {
							e.printStackTrace();
						} finally {
							callBack.onSuccess();
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
								System.out.println(Build.MODEL + " :: " + device.getString("name"));
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

	public void play(Song song, int position){
		String endpoint = "https://api.spotify.com/v1/me/player/play";
		String[] songUri = new String[]{song.getUri()};
		String pos = String.valueOf(position);

		Map<String, String> params = new HashMap<>();
		params.put("uris", songUri.toString());
		params.put("position_ms", pos);

		JSONObject parameters = new JSONObject(params);

		JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
				Request.Method.PUT,
				endpoint,
				parameters,
				new Response.Listener<JSONObject>() {
					@Override
					public void onResponse(JSONObject response) {

					}
				}, new Response.ErrorListener() {
			@Override
			public void onErrorResponse(VolleyError error) {

			}
		}){
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

	public void play(ArrayList<Song> songs) {
		String endpoint = "https://api.spotify.com/v1/me/player/play";
		ArrayList<String> uriList = new ArrayList<>();
		for(int i = 0; i < songs.size(); i++) {
			uriList.add(songs.get(i).getUri());
		}
		Map<String, String[]> uris = new HashMap<>();
		uris.put("uris",uriList.toArray(new String[uriList.size()]));

		JSONObject params = new JSONObject(uris);
		String device = "device_id=" + deviceId;

		endpoint = endpoint + "?" + device;
		JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
				Request.Method.PUT,
				endpoint,
				params,
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
		}) {
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
		}) {
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
		}) {
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
