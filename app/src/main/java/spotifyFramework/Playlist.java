package spotifyFramework;

import android.graphics.Bitmap;
import java.util.List;

public class Playlist {

	private List<Song> liked_songs;
	private List<Song> disliked_songs;
	private String title;
	private String description;
	private Bitmap cover;

	public Playlist(String title, String desc, List<Song> lsongs, List<Song> dsongs, Bitmap cover) {
		liked_songs = lsongs;
		disliked_songs = dsongs;
		this.title = title;
		description = desc;
		this.cover = cover;
	}

	public Playlist() {
	}

	public Bitmap getCover() {
		return cover;
	}

	public void setCover(Bitmap cover) {
		this.cover = cover;
	}

	public String getDescription() {
		return description;
	}

	public void setDescription(String description) {
		this.description = description;
	}

	public String getTitle() {
		return title;
	}

	public void setTitle(String title) {
		this.title = title;
	}

	public List<Song> getDisliked_songs() {
		return disliked_songs;
	}

	public void setDisliked_songs(List<Song> disliked_songs) {
		this.disliked_songs = disliked_songs;
	}

	public List<Song> getLiked_songs() {
		return liked_songs;
	}

	public void setLiked_songs(List<Song> liked_songs) {
		this.liked_songs = liked_songs;
	}
}
