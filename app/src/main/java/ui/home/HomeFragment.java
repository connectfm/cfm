package ui.home;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.cfm.recommend.Recommender;
import com.example.cfm.R;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import spotifyFramework.Song;
import spotifyFramework.SongService;

public class HomeFragment extends Fragment {

	public SongService songService;
	private RecyclerView recyclerView;
	private SongsAdapter adapter;
	private Recommender recommender;
	private Button radioStart;
	private SharedPreferences preferences;

	public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
		preferences = getActivity().getSharedPreferences("SPOTIFY", 0);
		View root = inflater.inflate(R.layout.fragment_home, container, false);
		radioStart = (Button) root.findViewById(R.id.begin_listening);
		songService = new SongService(getActivity());
		songService.getRecentlyPlayed(() -> {
			ArrayList<Song> recents = songService.getPlaylist();
			adapter = new SongsAdapter(getActivity(), recents);
			adapter.notifyDataSetChanged();
			recyclerView = (RecyclerView) getActivity().findViewById(R.id.listening_history);
			recyclerView.setAdapter(adapter);
			recyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
		});

        radioStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
					NavController navController = Navigation
							.findNavController(getActivity(), R.id.nav_host_fragment);
					navController.navigate(R.id.song_dashboard);
			}
		});

		return root;
	}
}