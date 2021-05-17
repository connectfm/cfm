package com.cfm.recommend;

import android.content.Context;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import lombok.Builder;
import lombok.Value;

import com.android.volley.toolbox.JsonRequest;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.google.gson.JsonObject;
import com.spotifyFramework.Song;
import com.spotifyFramework.SongService;
import com.spotifyFramework.VolleyCallBack;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Provides song recommendations in the form of song URIs.
 */
public class Recommender {
	private SongService songService;
	private final RequestQueue queue;
	private List<String> songs;
	private String recommendation;
	public Recommender(Context context) {
		songs = new ArrayList<>();
		queue = Volley.newRequestQueue(context);
		songService = new SongService(context);
	}

	public String getRecommenation() {return recommendation;}
	public List<String> getAllSongs() {return songs;}

	public void get(String id,final VolleyCallBack callBack) {
		String endpoint = "https://8vjxa5x5bc.execute-api.us-east-2.amazonaws.com/dev/recommendation";
		try {
			JSONObject params = new JSONObject();
			params.put("id", id);
			JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
					Request.Method.POST,
					endpoint,
					params,
					new Response.Listener<JSONObject>() {
						@Override
						public void onResponse(JSONObject response) {
							try {
								recommendation = response.getString("message");
								songs.add(response.getString("message"));
							} catch (JSONException e) {
								e.printStackTrace();
							}
							callBack.onSuccess();
						}
					},
					new Response.ErrorListener() {
						@Override
						public void onErrorResponse(VolleyError error) {
							System.out.println(error.toString());
						}
					}
			);
			int timeout = 25000; // 25 seconds

			jsonObjectRequest.setRetryPolicy(new DefaultRetryPolicy(
					timeout,
					DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
					DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));

			queue.add(jsonObjectRequest);
		} catch (JSONException e) {
			e.printStackTrace();
		}

	}

}
