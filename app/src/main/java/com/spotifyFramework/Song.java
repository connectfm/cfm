package com.spotifyFramework;

import com.google.gson.Gson;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Song {

	private String id;
	private String uri;
	private String name;
	private Long duration_ms;
	private Map<String,String> artistsList = new HashMap<String,String>();
	private List<String> albumImages = new ArrayList<String>();
	private String albumName;
	private int status;

	//Song Features Fields
	private float danceability;
	private float energy;
	private float loudness;
	private float speechiness;
	private float acousticness;
	private float instrumentalness;
	private float liveness;
	private float valence;
	private float tempo;

	public Song(String id, String name) {
		this.name = name;
		this.id = id;
		status = 2;
	}

	public String getId() {

		if(id==null)
			id = uri.substring(uri.lastIndexOf(":"),uri.length());
		return id;
	}

	public String getName() {
		return name;
	}

	public String getAlbum() {
		return albumName;
	}

	public Map<String, String> getArtists() {
		return artistsList;
	}

	public List<String> getArtistNames() {
		List<String> artists = new ArrayList<>();
		for(String s: artistsList.keySet()) {
			artists.add(s);
		}
		return artists;
	}
	public String getUri() {
		return uri;
	}

	public Long getDuration() {
		return duration_ms;
	}

	public int getStatus() {return status;}

	public float getAcousticness() {
		return acousticness;
	}

	public float getDanceability() {
		return danceability;
	}

	public float getEnergy() {
		return energy;
	}

	public float getInstrumentalness() {
		return instrumentalness;
	}

	public float getLiveness() {
		return liveness;
	}

	public float getLoudness() {
		return loudness;
	}

	public float getSpeechiness() {
		return speechiness;
	}

	public float getTempo() {
		return tempo;
	}

	public float getValence() {
		return valence;
	}

	public void setFeatures(JSONObject features) throws JSONException {

		acousticness = features.getLong("acousticness");
		danceability = features.getLong("danceability");
		energy = features.getLong("energy");
		loudness = features.getLong("loudness");
		speechiness = features.getLong("speechiness");
		acousticness = features.getLong("instrumentalness");
		liveness = features.getLong("liveness");
		valence = features.getLong("valence");
		tempo = features.getLong("tempo");
	}

	public void setAlbumName(String name) {
		this.albumName = name;
	}


	public void setTitle() {
		this.name = name;
	}

	public List<String> getImages() {
		return albumImages;
	}

	public void setImage(String image) {
		if (albumImages == null) {
			albumImages = new ArrayList<String>();
		}
		albumImages.add(image);
	}

	public void setArtist(String artist, String artistId) {
		if(artistsList == null) {
			artistsList = new HashMap<String, String>();
		}
		artistsList.put(artistId, artist);
	}

	public void setStatus(int x) {
		if(x <= 3 && x >=1)
			status = x;
	}

	public String toString() {
		Gson gson = new Gson();
		String res = gson.toJson(this);
		return res;
	}

	public String artistsToString(int limit) {
		StringBuilder sb = new StringBuilder();
		sb.append(getArtistNames().get(0));
		for (int i = 1; i < getArtists().size(); i++) {
			sb.append(", " + getArtists().get(i));
		}

		String res = sb.toString();
		if (res.length() > limit) {
			res = res.substring(0, limit) + "...";
		}
		return res;
	}



}
