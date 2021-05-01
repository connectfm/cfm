package ui.databases;

import android.app.Application;
import androidx.lifecycle.LiveData;
import java.util.List;
import spotifyFramework.Playlist;

public class PlaylistRepo {

	private LiveData<List<Playlist>> playlists;

	public PlaylistRepo(Application app) {
	}

	public PlaylistRepo() {
	}

	public void insert(Playlist p) {
	}

	public void delete(Playlist p) {
	}

	public void update(Playlist p) {
	}

	public LiveData<List<Playlist>> getAllPlaylists() {
		return playlists;
	}

	public void setAllPlaylists(List<Playlist> plists) {
	}

}
