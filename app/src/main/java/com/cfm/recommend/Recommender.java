package com.cfm.recommend;

import androidx.core.util.Supplier;

import com.android.volley.Request;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;

import org.json.JSONObject;

import java.util.concurrent.Callable;
import lombok.Builder;
import lombok.Value;
import spotify_framework.VolleyCallBack;

/**
 * Provides song recommendations in the form of song URIs.
 */
@Value
@Builder
public class Recommender {

	public void get( final VolleyCallBack callBack) {
		String endpoint = "https://whateverourURLis.com/";
		JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
				Request.Method.GET,
				endpoint,
				null,
				new Response.Listener<JSONObject>() {
					@Override
					public void onResponse(JSONObject response) {
						/*
						We'd get the call, and probably call the Spotify API for info
						On the URIs (I've already got a method set up for it in SongService)

						Then we store the now Song data structures in a List within this object

						After all that, we call the callback bc we know the data is compiled and
						properly stored
						 */
						callBack.onSuccess();

						/*
						Then we can do whatever we want down here to give info back to the server
						abt updated information (We could have prior info as a param)
						 */
					}
				}, new Response.ErrorListener() {
			@Override
			public void onErrorResponse(VolleyError error) {

			}
		}
		);
		// Connect to the recommendation system -- may be able to do this through Amplify
		// Request a recommendation in the form of a song URI
	}
	
}
