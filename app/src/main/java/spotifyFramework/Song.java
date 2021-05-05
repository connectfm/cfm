package spotifyFramework;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class Song {

	private final String id;
	private String uri;
	private String name;
	private Long duration_ms;
	private List<String> artistsList = new ArrayList<String>();
	private List<String> albumImages = new ArrayList<String>();
	private String artist_id;
	private String albumName;

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
	}

	public String getId() {
		return id;
	}

	public String getName() {
		return name;
	}

	public String getAlbum() {
		return albumName;
	}

	public List<String> getArtists() {
		return artistsList;
	}

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

	public void setArtist(String artist) {
		if (artistsList == null) {
			artistsList = new ArrayList<String>();
		}
		artistsList.add(artist);
	}

	public String toString() {
		Gson gson = new Gson();
		String res = gson.toJson(this);
		return res;
	}

	public String artistsToString(int limit) {
		StringBuilder sb = new StringBuilder();
		sb.append(getArtists().get(0));
		for (int i = 1; i < getArtists().size(); i++) {
			sb.append(", " + getArtists().get(i));
		}

		String res = sb.toString();
		if (res.length() > limit) {
			res = res.substring(0, limit) + "...";
		}
		return res;
	}

	public String getUri() {
		return uri;
	}

	public Long getDuration() {
		return duration_ms;
	}

}
